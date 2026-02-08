import asyncio
from typing import Any
from textual import events
from textual.widget import Widget
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static, RadioSet, RadioButton


class RadiolistPopup(Widget):
    """
    A popup widget that displays a scrollable radio list for selection and returns the chosen valuei.
    """

    def __init__(
        self, title: str, text: str, options: list[tuple[str, str]], **kwargs: Any
    ) -> None:
        """
        :param title: The title of the popup.
        :param text: The prompt text.
        :param options: A list of (option_value, label) tuples.
        """
        super().__init__(**kwargs)
        self.title = title
        self.text = text
        self.options = options
        self.choice_future: asyncio.Future = asyncio.Future()

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]{self.title}[/bold]\n{self.text}\n", classes="full-width")
        with ScrollableContainer(id="popup_scroll_view", classes="full-width"):
            with RadioSet(id="popup_radio_set", classes="full-width"):
                for _, label in self.options:
                    yield RadioButton(label, classes="full-width")

    async def on_mount(self) -> None:
        self.query_one(RadioSet).focus()
        self.focus()

    async def on_key(self, event: events.Key) -> None:
        radio_set = self.query_one(RadioSet)
        index = radio_set.pressed_index
        if event.key in ("enter", "space"):
            if index != -1:
                choice = self.options[index][0]
                if not self.choice_future.done():
                    self.choice_future.set_result(choice)
        elif event.key == "escape":
            if not self.choice_future.done():
                self.choice_future.set_result("cancel")
        elif event.key in ("up", "down"):
            selected_button = (
                radio_set.children[index]
                if 0 <= index < len(radio_set.children)
                else None
            )
            if selected_button:
                scroll_view = self.query_one("#popup_scroll_view")
                scroll_view.scroll_to_widget(selected_button, animate=True)
        await self.focus_self()

    async def focus_self(self) -> None:
        self.focus()

    async def wait_for_choice(self) -> str:
        return await self.choice_future
