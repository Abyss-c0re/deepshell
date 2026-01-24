import os
import re
import asyncio
import numpy as np
from datetime import datetime
from utils.logger import Logger
from typing import Tuple, Optional
from prompts.helper import PromptHelper
from utils.file_utils import _read_file
from ollama_client.api_client import OllamaClient
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import OFF_THR, MSG_THR, CONT_THR, NUM_MSG, OFF_FREQ, SLICE_SIZE

logger = Logger.get_logger()


class Project:
    def __init__(self, name: str = "") -> None:
        self.name: str = name
        self.file_embeddings: dict[str, dict] = {}
        self.folder_structure: dict = {}

    def _index_content(
        self,
        identifier: str,
        content: str,
        embedding: np.ndarray,
        content_type: str = "file",
    ) -> None:
        """
        Generic method to index any content (files or terminal outputs).

        Args:
            identifier (str): Unique identifier (e.g., file path or generated key).
            content (str): The content to index.
            embedding (np.ndarray): The computed embedding.
            content_type (str): Type of the content ("file" or "terminal").
        """
        content_info = {
            "identifier": identifier,
            "content": content,
            "embedding": embedding,
            "type": content_type,
        }
        self.file_embeddings[identifier] = content_info
        logger.debug(
            f"Project '{self.name}': Added {content_type} content with id {identifier}"
        )

    def _index_file(self, file_path: str, content: str, embedding: np.ndarray) -> None:
        """
        Indexes a file's embedding, wrapping the file path as the unique identifier.
        """
        self._index_content(file_path, content, embedding, content_type="file")

    def _index_terminal_output(
        self, output: str, identifier: str, embedding: np.ndarray
    ) -> None:
        """
        Indexes terminal code blocks or output, generating a unique identifier if not provided.
        """
        if not identifier:
            # Generate a unique identifier, e.g., using a timestamp.
            identifier = f"terminal_{datetime.now().isoformat()}"
        self._index_content(identifier, output, embedding, content_type="terminal")


class Topic:
    def __init__(self, name: str = "", description: str = "") -> None:
        """
        Initializes a Topic with a name and description.
        The description is embedded and cached for matching.

        Args:
            name (str): The topic name.
            description (str): A textual description of the topic.
        """
        self.name = name
        self.description = description
        self.embedded_description = np.array([])
        self.history: list[dict[str, str]] = []
        self.history_embeddings = []
        self.embedding_cache: dict[str, np.ndarray] = {}

    async def add_message(self, role: str, message: str, embedding: np.ndarray) -> None:
        """
        Stores raw messages and their embeddings.
        """
        self.history.append({"role": role, "content": message})
        self.history_embeddings.append(embedding)
        logger.info(f"Message added to: {self.name}")

    async def get_relevant_context(self, embedding: np.ndarray) -> tuple[float, int]:
        """
        Retrieves the best similarity score and the index of the most relevant message
        from the topic’s history based on cosine similarity.

        Args:
            query_embedding.
            similarity_threshold (float): The base similarity threshold.

        Returns:
            tuple[float, int]: A tuple containing:
                - The best similarity score.
                - The index of the best matching message (or -1 if not found).
        """
        if not self.history_embeddings:
            logger.info("No history embeddings found. Returning empty context.")
            return 0.0, -1

        similarities = cosine_similarity([embedding], self.history_embeddings)[0]
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        logger.debug(f"Best similarity score: {best_similarity} at index {best_index}")

        return best_similarity, best_index


class HistoryManager:
    def __init__(self, manager) -> None:
        """
        Initializes HistoryManager to handle topics and off-topic tracking.
        An "unsorted" topic collects messages and files until a clear topic emerges.

        Args:
            top_k (int): Maximum number of context items to retrieve.
            similarity_threshold (float): Threshold for determining similarity.
        """
        self.file_utils = manager.file_utils
        self.helper = manager._handle_helper_mode
        self.tasker = manager.deploy_chatbot_method
        self.ui = manager.ui
        self.similarity_threshold = MSG_THR
        self.topics: list[Topic] = []
        self.current_topic = Topic("Initial topic")
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.projects: list[Project] = []
        self.current_project = Project("Unsorted")

    async def add_message(
        self, role: str, message: str, embedding: np.ndarray | None = None
    ) -> None:
        """
        Routes a new message to the best-matching topic.
        If no topic meets the similarity threshold, the message is added to the unsorted topic.

        Args:
            role (str): Sender's role.
            message (str): The message text.
        """
        if not embedding:
            embedding = await self.fetch_embedding(message)
        topic = await self._match_topic(embedding, exclude_topic=self.current_topic)
        if topic:
            await self.switch_topic(topic)

        await self.current_topic.add_message(role, message, embedding)
        asyncio.create_task(self._analyze_history())

    async def add_file(
        self, file_path: str, content: str, folder: bool = False
    ) -> None:
        """
        Adds a file by computing its combined embedding (file path + content)
        and routing it to the appropriate project based on the file's folder.
        If the file's project folder (extracted from the file path) is different
        from the current project's name, the current project is archived and a new
        project is created and assigned.
        """
        new_project_name = os.path.basename(os.path.dirname(file_path))

        if self.current_project.name.lower() != new_project_name.lower():
            if (
                self.current_project.name.lower() != "unsorted"
                and self.current_project not in self.projects
            ):
                self.projects.append(self.current_project)
                logger.info(
                    f"Archived project '{self.current_project.name}' to projects list."
                )

            if not self.current_project.folder_structure and not folder:
                if await self.ui.yes_no_prompt(
                    "Do you want to generate structure for this file's folder?", "No"
                ):
                    new_project = Project(new_project_name)
                    try:
                        folder_path = os.path.dirname(file_path)
                        structure = self.file_utils.generate_structure(
                            folder_path, folder_path
                        )
                        new_project.folder_structure = structure
                        logger.info(
                            f"Generated new folder structure for project '{new_project_name}'."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate structure for project '{new_project_name}': {e}"
                        )

                    self.current_project = new_project

        # Compute embedding for file path + content
        combined_content = f"Path: {file_path}\nContent: {content}"
        file_embedding = await self.fetch_embedding(combined_content)

        # Store the file in the project using a universal indexing method
        self.current_project._index_content(
            file_path, content, file_embedding, content_type="file"
        )

    async def add_terminal_output(
        self, command: str, output: str, summary: str
    ) -> None:
        """
        Adds a terminal output by computing its embedding and indexing it
        within the current project.

        Args:
            command (str): The executed command.
            output (str): The raw output of the command.
            summary (str): A summarized explanation of the output.
        """
        terminal_content = f"Command: {command}\nOutput: {output}\nSummary: {summary}"
        terminal_embedding = await self.fetch_embedding(terminal_content)

        # Generate a unique identifier for terminal output storage
        terminal_id = f"terminal_{hash(command + datetime.now().isoformat())}"

        # Store the terminal output using the unified indexing method
        self.current_project._index_content(
            terminal_id, terminal_content, terminal_embedding, content_type="terminal"
        )

        logger.info(f"Stored terminal output for command: {command}")

    def add_folder_structure(self, structure: dict) -> None:
        """
        Adds or updates folder structure for the current project.
        If a structure already exists, archives the current project by adding it
        to the projects list (if not already present) before updating.
        Also, if the current project's name is empty or 'Unsorted', assigns the new folder name.
        """
        if self.current_project.folder_structure:
            if self.current_project not in self.projects:
                self.projects.append(self.current_project)
                logger.info(
                    f"Archived project '{self.current_project.name}' to projects list."
                )

        if (
            not self.current_project.name
            or self.current_project.name.lower() == "unsorted"
        ):
            if isinstance(structure, dict) and len(structure) == 1:
                new_name = list(structure.keys())[0]
                self.current_project.name = new_name
                logger.info(
                    f"Assigned new project name '{new_name}' from folder structure."
                )

        self.current_project.folder_structure = structure
        logger.info(
            f"Folder structure updated for project '{self.current_project.name}'."
        )

    def format_structure(self, folder_structure: dict) -> str:
        """
        Formats the folder structure dictionary into a readable string format.
        """

        def format_substructure(substructure, indent=0):
            formatted = ""
            for key, value in substructure.items():
                if isinstance(value, dict):  # Subfolder
                    formatted += " " * indent + f"{key}/\n"
                    formatted += format_substructure(value, indent + 4)
                else:
                    formatted += " " * indent + f"-- {value}\n"
            return formatted

        return format_substructure(folder_structure)

    def find_project_structure(self, query: str) -> Project | None:
        """
        Checks if the query contains a folder name corresponding to one of the existing projects.
        Returns the matching Project if found.
        """
        for project in self.projects:
            if project.name.lower() in query.lower():
                logger.info(f"Found project structure for '{project.name}' in query")
                return project
        logger.info("No matching project found in query")
        return None

    def extract_file_name_from_query(self, query: str) -> str | None:
        file_pattern = r"([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)*/[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)"
        match = re.search(file_pattern, query)
        if match:
            return match.group(0)
        return None

    def extract_folder_from_query(self, query: str) -> str | None:
        """
        Extracts a folder path from the query.
        This regex looks for paths containing at least one slash and that do not end with an extension.
        """
        folder_pattern = r"([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)+)"
        match = re.search(folder_pattern, query)
        if match:
            candidate = match.group(0)
            if not re.search(r"\.[a-zA-Z0-9]+$", candidate):
                return candidate
        return None

    async def get_relevant_content(
        self,
        query: str,
        content_type: Optional[str] = None,
        top_k: int = 1,
        similarity_threshold: float = CONT_THR,
    ) -> list | None:
        """
        Retrieves relevant content (files or terminal outputs) by comparing the query
        against the stored embeddings.

        Args:
            query (str): The user query.
            content_type (str, optional): Type of content to filter by (e.g., "file" or "terminal").
            top_k (int): Maximum number of results.
            similarity_threshold (float): Minimum similarity score to consider.

        Returns:
            list: A list of tuples (identifier, content) for the top matching entries.
        """
        query_embedding = await self.fetch_embedding(query)
        scores = []

        # Optionally extract a file name only if querying for files.
        file_name = (
            self.extract_file_name_from_query(query)
            if content_type in (None, "file")
            else None
        )

        # Iterate over all stored content
        for identifier, info in self.current_project.file_embeddings.items():
            # Filter by type if specified
            if content_type and info.get("type") != content_type:
                continue

            # If querying a file, try an exact match on identifier if available.
            if file_name and info.get("type") == "file":
                if file_name.lower() in info.get("identifier", "").lower():
                    scores.append((identifier, 1.0))
                    logger.info(
                        f"Added file '{identifier}' to context (Exact match on file name)."
                    )
                    continue

            similarity = cosine_similarity([query_embedding], [info["embedding"]])[0][0]
            if similarity >= similarity_threshold:
                scores.append((identifier, similarity))
                logger.info(
                    f"Added file '{identifier}' to context (Similarity: {similarity})."
                )

        # Expand search to other projects if necessary (similar to your current logic)
        if not scores:
            logger.info(
                "No relevant content in the current project; searching across all projects."
            )
            for project in self.projects:
                for identifier, info in project.file_embeddings.items():
                    if content_type and info.get("type") != content_type:
                        continue
                    similarity = cosine_similarity(
                        [query_embedding], [info["embedding"]]
                    )[0][0]
                    if similarity >= similarity_threshold:
                        scores.append((identifier, similarity))
                        logger.info(
                            f"Added file '{identifier}' from project '{project.name}' to context (Similarity: {similarity})."
                        )

        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            selected_ids = [id for id, _ in scores[:top_k]]
            results = []
            for id in selected_ids:
                # Try to retrieve the content from the current project first.
                if (
                    id in self.current_project.file_embeddings
                    and "content" in self.current_project.file_embeddings[id]
                ):
                    results.append(
                        (id, self.current_project.file_embeddings[id]["content"])
                    )
                    logger.info(f"Added content from file '{id}' to results.")
                else:
                    content = await _read_file(id)
                    results.append((id, content))
                    logger.info(
                        f"Added content from file '{id}' to results (Read from file path)."
                    )
            return results

        logger.info("No matching content found.")
        return None

    async def fetch_embedding(self, text: str) -> np.ndarray:
        """
        Asynchronously fetches and caches an embedding for the given text.
        Uses async lock to guard the caching mechanism.
        """
        async with asyncio.Lock():
            if text in self.embedding_cache:
                return self.embedding_cache[text]

        embedding = await self.tasker(OllamaClient.fetch_embedding, text)
        if embedding:
            self.embedding_cache[text] = embedding
            logger.debug(f"Extracted {len(embedding)} embeddings")
            return embedding
        else:
            return np.array([])

    async def switch_topic(self, topic: Topic) -> None:
        async with asyncio.Lock():
            if topic.name != self.current_topic.name:
                if not any(t.name == self.current_topic.name for t in self.topics):
                    self.topics.append(self.current_topic)
                logger.info(f"Switched to {topic.name}")
                self.current_topic = topic

    async def _match_topic(
        self, embedding: np.ndarray, exclude_topic: Topic | None = None
    ) -> Topic | None:
        """
        Matches a message or file embedding to the most similar topic based on the description embedding,
        optionally excluding a specified topic.

        Args:
            embedding (np.ndarray): The embedding to match.
            exclude_topic (Topic | None): A topic to exclude from matching (e.g. the current topic).

        Returns:
            Topic | None: The best matching topic if similarity exceeds the threshold.
        """
        if len(self.topics) == 0:
            logger.info("No topics available for matching. Returning None.")
            return None

        async def compute_similarity(topic: Topic) -> tuple[float, Topic]:
            if len(topic.embedded_description) == 0 or len(embedding) == 0:
                return 0.0, topic
            similarity = cosine_similarity([embedding], [topic.embedded_description])[
                0
            ][0]
            logger.debug(
                f"Computed similarity {similarity:.4f} for topic '{topic.name}'"
            )
            return similarity, topic

        tasks = [
            compute_similarity(topic) for topic in self.topics if topic != exclude_topic
        ]
        results = await asyncio.gather(*tasks)

        best_topic = None
        best_similarity = 0.0
        for similarity, topic in results:
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_topic = topic

        if best_topic:
            logger.info(
                f"Best matching topic: '{best_topic.name}' with similarity {best_similarity:.4f}"
            )
            return best_topic
        else:
            logger.info("No suitable topic found.")
            return None

    async def generate_prompt(self, query: str, num_messages: int = NUM_MSG) -> list:
        """
        Generates a prompt by retrieving context and content references (files, terminal outputs, etc.)
        from the best matching topic. If the query references a folder, the corresponding folder structure
        is retrieved and assigned to the current topic before being included in the prompt.

        Args:
            query (str): The user query.

        Returns:
            list: The last few messages from the topic's history.
        """
        embedding = await self.fetch_embedding(query)

        # Determine the best matching topic and switch to it if found.
        current_topic = await self._match_topic(embedding)
        if current_topic:
            await self.switch_topic(current_topic)

        # Retrieve the project folder structure if the query contains a folder name/path.
        project = self.find_project_structure(query)
        if project:
            self.current_project = project

        # Retrieve all types of content unless a filter is specified.
        relevant_content = await self.get_relevant_content(query)
        content_references = ""

        if relevant_content:
            if self.current_project.folder_structure:
                content_references += f"Folder structure:\n{self.format_structure(self.current_project.folder_structure)}\n"
            # Iterate through each retrieved content item.
            for identifier, content in relevant_content:
                # Attempt to determine the content type (default to generic "Content").
                content_type = self.current_project.file_embeddings.get(
                    identifier, {}
                ).get("type", "content")
                if content_type == "file":
                    label = "Referenced File"
                elif content_type == "terminal":
                    label = "Referenced Terminal Output"
                else:
                    label = "Referenced Content"
                content_references += f"\n[{label}: {identifier}]\n{content}\n"

        prompt = (
            f"{content_references}\nUser query: {query}"
            if content_references
            else query
        )

        logger.debug(f"Generated prompt: {prompt}")
        await self.add_message("user", prompt, embedding)

        return self.current_topic.history[-num_messages:]

    async def generate_topic_info_from_history(
        self, history: list, max_retries: int = 3
    ) -> Tuple[str, str] | Tuple[None, None]:
        """
        Attempts to extract a topic name and description from the given history.
        :
        Args:
            history (list): List of unsorted history messages.
            max_retries (int): Maximum number of attempts.

        Returns:
            tuple: (extracted_topic_name, extracted_topic_description) if successful; otherwise (None, None).
        """
        attempt = 0
        extracted_topic_name = None
        extracted_topic_description = None

        while attempt < max_retries:
            await asyncio.sleep(1)
            try:
                response = await self.helper(PromptHelper.topics_helper(history), True)
                if not response:
                    logger.warning("Received empty response from the helper.")

                logger.debug(f"Extracting topic info from response: {response}")
                clean_response = re.sub(
                    r"^```|```$|json", "", response, flags=re.IGNORECASE
                ).strip()
                matches = re.findall(r':\s*"([^"]+)"', clean_response)
                extracted_topic_name = matches[0] if len(matches) > 0 else "unknown"
                extracted_topic_description = matches[1] if len(matches) > 1 else ""

                if extracted_topic_name and extracted_topic_description:
                    logger.info(f"Extracted topic: {extracted_topic_name}")
                    return extracted_topic_name, extracted_topic_description
                else:
                    logger.warning("Could not extract valid topic information.")
            except Exception as e:
                logger.error(
                    f"Analyze history attempt {attempt + 1} failed: {str(e)}",
                    exc_info=True,
                )
                attempt += 1
                if attempt < max_retries:
                    logger.info(
                        f"Retrying analysis... (Attempt {attempt + 1} of {max_retries})"
                    )
                else:
                    logger.error(
                        "Max analysis retries reached; not splitting unsorted history."
                    )
                    break
        return None, None

    async def _analyze_history(
        self,
        off_topic_threshold: float = OFF_THR,
        off_topic_frequency: int = OFF_FREQ,
        slice_size: int = SLICE_SIZE,
    ) -> None:
        """
        Analyzes the current topic's history for potential off-topic drift.
        When the history length reaches a multiple of `off_topic_frequency`, the method:
          1. Takes a slice of the last `slice_size` messages and computes per-message similarity to the current topic.
          2. If more than half of the messages in the slice have a similarity below `off_topic_threshold`,
             it determines the precise start of the off-topic segment.
          3. Generates candidate topic info for the off-topic segment and attempts to match it with an existing topic
             (excluding the current topic). If a match is found, the off-topic messages are reassigned; otherwise,
             a new topic is created.
          4. Finally, the off-topic messages are removed from the current topic.
        """
        # If the current topic is unnamed but has > 4 messages, generate a topic name/description.
        if (
            len(self.current_topic.history) > 4
            and not self.current_topic.description.strip()
        ):
            (
                new_topic_name,
                new_topic_desc,
            ) = await self.generate_topic_info_from_history(self.current_topic.history)
            if new_topic_name and new_topic_desc:
                self.current_topic.name = new_topic_name
                self.current_topic.description = new_topic_desc
                self.current_topic.embedded_description = await self.fetch_embedding(
                    new_topic_desc
                )
                return

        # Trigger analysis when history length is a multiple of off_topic_frequency.
        if (
            len(self.current_topic.history) >= off_topic_frequency
            and len(self.current_topic.history) % off_topic_frequency == 0
        ):
            logger.info("Analyzing current topic for potential off-topic segments.")

            current_name = self.current_topic.name

            # Candidate slice: the last `slice_size` messages.
            candidate_slice = self.current_topic.history[-slice_size:]

            # Concurrently fetch embeddings for the candidate slice.
            candidate_embeddings = await asyncio.gather(
                *(self.fetch_embedding(msg["content"]) for msg in candidate_slice)
            )

            similarities = []
            for msg_emb in candidate_embeddings:
                sim = cosine_similarity(
                    [msg_emb], [self.current_topic.embedded_description]
                )[0][0]
                similarities.append(sim)
            logger.info(f"Per-message similarities for candidate slice: {similarities}")

            # Check if more than half of the messages fall below the threshold.
            if (
                sum(1 for s in similarities if s < off_topic_threshold)
                > len(similarities) / 2
            ):
                # Identify the precise start index of the off-topic segment.
                off_topic_start_index = len(self.current_topic.history) - slice_size
                for i, sim in enumerate(similarities):
                    if sim < off_topic_threshold:
                        off_topic_start_index = (
                            len(self.current_topic.history) - slice_size + i
                        )
                        break
                off_topic_segment = self.current_topic.history[off_topic_start_index:]
                logger.info(
                    f"Identified off-topic segment from index {off_topic_start_index} to end "
                    f"(total {len(off_topic_segment)} messages)."
                )

                # Generate candidate topic info from the off-topic segment.
                (
                    candidate_topic_name,
                    candidate_topic_desc,
                ) = await self.generate_topic_info_from_history(off_topic_segment)
                if candidate_topic_name and candidate_topic_desc:
                    candidate_embedding = await self.fetch_embedding(
                        candidate_topic_desc
                    )
                    matched_topic = await self._match_topic(
                        candidate_embedding, exclude_topic=self.current_topic
                    )
                    if matched_topic is not None:
                        logger.info("Matched topic found")
                        # Reassign off-topic messages to the matched topic.
                        for msg in off_topic_segment:
                            msg_emb = await self.fetch_embedding(msg["content"])

                            await matched_topic.add_message(
                                msg["role"], msg["content"], msg_emb
                            )
                        logger.info(
                            f"Reassigned off-topic segment of {len(off_topic_segment)} messages to existing topic "
                            f"'{matched_topic.name}'."
                        )
                        await self.switch_topic(matched_topic)

                    else:
                        # No matching topic found—create a new topic.
                        try:
                            logger.info("Creating new topic from the off-topic content")
                            new_topic = Topic(
                                candidate_topic_name, candidate_topic_desc
                            )
                            new_topic.embedded_description = candidate_embedding
                            for msg in off_topic_segment:
                                msg_emb = await self.fetch_embedding(msg["content"])

                                await new_topic.add_message(
                                    msg["role"], msg["content"], msg_emb
                                )

                            await self.switch_topic(new_topic)

                            logger.info(
                                f"Created new topic '{new_topic.name}' with {len(off_topic_segment)} off-topic messages."
                            )

                        except Exception as e:
                            logger.error(
                                f"Error creating new topic from off-topic messages: {e}",
                                exc_info=True,
                            )

                        target_topic = next(
                            (
                                topic
                                for topic in self.topics
                                if topic.name == current_name
                            ),
                            None,
                        )
                        if target_topic:
                            async with asyncio.Lock():
                                target_topic.history = target_topic.history[
                                    :off_topic_start_index
                                ]
                                logger.info("Removed off-topic from the current topic")

                else:
                    logger.warning(
                        "Could not generate candidate topic info from the off-topic segment."
                    )
            else:
                logger.info(
                    "Candidate slice does not appear off-topic; no splitting performed."
                )

        return
