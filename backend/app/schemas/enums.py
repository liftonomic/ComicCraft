"""Shared enums for the comic generation lifecycle."""

from __future__ import annotations

from enum import Enum


class GenerationStatus(str, Enum):
    """State machine for a comic generation job.

    Normally a comic advances ``CREATED -> GENERATING_STORY ->
    GENERATING_IMAGES -> BUILDING_COMIC -> READY``. Any state can transition
    to ``FAILED`` when a stage raises an error.
    """

    CREATED = "CREATED"
    GENERATING_STORY = "GENERATING_STORY"
    GENERATING_IMAGES = "GENERATING_IMAGES"
    BUILDING_COMIC = "BUILDING_COMIC"
    READY = "READY"
    FAILED = "FAILED"


#: Ordered progress milestones (percentage) for the happy path.
NORMAL_SEQUENCE: list[GenerationStatus] = [
    GenerationStatus.CREATED,
    GenerationStatus.GENERATING_STORY,
    GenerationStatus.GENERATING_IMAGES,
    GenerationStatus.BUILDING_COMIC,
    GenerationStatus.READY,
]

#: Progress when entering each normal stage. Used to render the progress bar.
STAGE_PROGRESS: dict[GenerationStatus, int] = {
    GenerationStatus.CREATED: 5,
    GenerationStatus.GENERATING_STORY: 25,
    GenerationStatus.GENERATING_IMAGES: 55,
    GenerationStatus.BUILDING_COMIC: 90,
    GenerationStatus.READY: 100,
    GenerationStatus.FAILED: 100,
}