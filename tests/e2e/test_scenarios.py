"""Provider-boundary tests, independent of Home Assistant/browser health."""

import asyncio
import unittest
from types import SimpleNamespace

from scenario_app import ScenarioAgent


class Scenarios(unittest.IsolatedAsyncioTestCase):
    def agent(self):
        return ScenarioAgent(
            SimpleNamespace(tool_registry=None, output_modalities={"text", "audio"})
        )

    async def test_audio_and_two_persistent_turns(self):
        agent = self.agent()
        await agent.start()
        for turn in ["one", "two"]:
            await agent.submit_audio(turn, b"\x01\x00" * 160)
            await agent.end_turn(turn)
            events = []
            async with asyncio.timeout(2):
                async for event in agent.events():
                    events.append(event)
                    if event.type == "assistant.response_finished":
                        break
            self.assertEqual(events[0].type, "assistant.response_started")
            chunks = [e for e in events if e.type == "assistant.audio.chunk"]
            self.assertEqual(sum(len(e.audio) for e in chunks), 28800)
            self.assertTrue(all(e.sample_rate == 24000 for e in chunks))
        await agent.close()
        self.assertTrue(agent._closed)
        self.assertIsNone(agent._response_task)

    async def test_browser_open_audio_turn_can_respond_without_terminal_marker(self):
        agent = self.agent()
        await agent.start()
        await agent.submit_audio("browser", b"\x01\x00" * 32000)
        events = []
        async with asyncio.timeout(2):
            async for event in agent.events():
                events.append(event.type)
                if event.type == "assistant.response_finished":
                    break
        self.assertIn("user.speech_started", events)
        self.assertIn("user.transcript.final", events)
        await agent.end_turn("browser")
        self.assertTrue(agent._events.empty())  # No duplicate response at detach.
        await agent.close()

    async def test_interrupt_and_close_cancel_delayed_response(self):
        agent = self.agent()
        await agent.start()
        await agent.submit_text("one", "slow")
        await agent.end_turn("one")
        first = await anext(agent.events())
        self.assertEqual(first.type, "assistant.response_started")
        await agent.interrupt()
        self.assertIsNone(agent._response_task)
        await agent.close()
        self.assertTrue(agent._closed)


if __name__ == "__main__":
    unittest.main()
