import logging

from gemynd.shared.logging_config import setup_logging


def test_setup_logging_hides_only_informational_neo4j_notifications() -> None:
    root = logging.getLogger()
    neo4j_notifications = logging.getLogger("neo4j.notifications")
    original_handlers = list(root.handlers)
    original_root_level = root.level
    original_notification_level = neo4j_notifications.level

    try:
        setup_logging("INFO")

        assert root.level == logging.INFO
        assert neo4j_notifications.getEffectiveLevel() == logging.WARNING
        assert neo4j_notifications.isEnabledFor(logging.INFO) is False
        assert neo4j_notifications.isEnabledFor(logging.WARNING) is True
    finally:
        root.handlers.clear()
        root.handlers.extend(original_handlers)
        root.setLevel(original_root_level)
        neo4j_notifications.setLevel(original_notification_level)
