import unittest

from streamlit.testing.v1 import AppTest


class AppModeTests(unittest.TestCase):
    def test_both_analysis_modes_render_without_exceptions(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("Competitor Analysis" in title.value for title in app.title))

        app.radio[0].set_value("IB style synthesis").run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("IB Style Synthesis" in title.value for title in app.title))
        self.assertTrue(any("Run IB synthesis" in button.label for button in app.button))


if __name__ == "__main__":
    unittest.main()
