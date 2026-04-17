#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/services/dataset_registry.py
# @IDE     : vscode

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Type
from uuid import NAMESPACE_URL, uuid5

from app.adapters.base import DatasetAdapter
from app.models import (
    DatasetCatalogEntry,
    DatasetCatalogResponse,
    DatasetConfig,
    DatasetSceneOption,
    SequenceSummary,
    SequenceTimelineResponse,
    TimelineFrame,
)


DEFAULT_DATASET_PLUGIN_ROOT = Path(__file__).resolve().parents[3] / "datasets"
DATASET_PLUGIN_ROOT = Path(os.getenv("DATASET_PLUGIN_ROOT", str(DEFAULT_DATASET_PLUGIN_ROOT)))


@dataclass
class SequenceState:
    sequence_id: str
    config: DatasetConfig
    adapter: DatasetAdapter


@dataclass
class DatasetDefinitionSpec:
    dataset_id: str
    name: str
    class_names: list[str]
    default_scene_id: str | None
    list_scenes: Callable[[], list[DatasetSceneOption]]
    build_config: Callable[[str], DatasetConfig]
    adapter_class: Type[DatasetAdapter]


class DatasetRegistry:
    def __init__(self) -> None:
        self._sequences: dict[str, SequenceState] = {}
        self._dataset_definitions = self._discover_dataset_definitions()

    def clear(self) -> None:
        self._sequences.clear()

    def build_adapter(self, config: DatasetConfig) -> DatasetAdapter:
        return self._build_adapter(config)

    def list_dataset_catalog(self) -> DatasetCatalogResponse:
        datasets = [self._catalog_entry_from_spec(spec) for spec in self._dataset_definitions.values()]
        default_dataset_id = "radelft" if "radelft" in self._dataset_definitions else (datasets[0].id if datasets else None)
        default_scene_id = None
        if default_dataset_id:
            default_dataset = next((dataset for dataset in datasets if dataset.id == default_dataset_id), None)
            if default_dataset:
                default_scene_id = self._choose_default_scene(default_dataset)
        return DatasetCatalogResponse(
            default_dataset_id=default_dataset_id,
            default_scene_id=default_scene_id,
            datasets=datasets,
        )

    def resolve_catalog_config(self, dataset_id: str, scene_id: str | None = None) -> DatasetConfig:
        spec = self._dataset_definitions.get(dataset_id)
        if spec is None:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        entry = self._catalog_entry_from_spec(spec)
        target_scene_id = scene_id or self._choose_default_scene(entry)
        if not target_scene_id:
            raise ValueError(f"Dataset {entry.name} does not expose any selectable scenes")
        scene = next((item for item in entry.scenes if item.id == target_scene_id), None)
        if scene is None:
            raise ValueError(f"Unknown scene {target_scene_id} for dataset {entry.name}")
        if not scene.available:
            raise ValueError(scene.reason or f"Scene {scene.name} is unavailable")
        return spec.build_config(scene.id)

    def load(self, *, config: DatasetConfig) -> SequenceState:
        if config is None:
            raise ValueError("Dataset config is required")

        sequence_id = self._sequence_id(config)
        if sequence_id in self._sequences:
            state = self._sequences[sequence_id]
            state.adapter.on_loaded()
            return state

        for existing in self._sequences.values():
            if (
                existing.config.adapter == config.adapter
                and existing.config.dataset_root == config.dataset_root
                and existing.config.sequence_name == config.sequence_name
            ):
                existing.adapter.on_loaded()
                return existing

        adapter = self._build_adapter(config)
        adapter.on_loaded()
        state = SequenceState(sequence_id=sequence_id, config=config, adapter=adapter)
        self._sequences[state.sequence_id] = state
        return state

    def sequence_id_for_config(self, config: DatasetConfig) -> str:
        return self._sequence_id(config)

    def _sequence_id(self, config: DatasetConfig) -> str:
        identity = "|".join(
            [
                config.adapter,
                config.sequence_name,
                config.dataset_root,
                config.calib_root or "",
                str(sorted(config.options.items())),
            ]
        )
        return str(uuid5(NAMESPACE_URL, identity))

    def _build_adapter(self, config: DatasetConfig) -> DatasetAdapter:
        spec = self._dataset_definitions.get(config.adapter)
        if spec is None:
            raise ValueError(f"Unsupported adapter: {config.adapter}")
        return spec.adapter_class(config)

    def _discover_dataset_definitions(self) -> dict[str, DatasetDefinitionSpec]:
        definitions: dict[str, DatasetDefinitionSpec] = {}
        if not DATASET_PLUGIN_ROOT.exists():
            return definitions
        for dataset_dir in sorted(DATASET_PLUGIN_ROOT.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
                continue
            dataset_module_path = dataset_dir / "dataset.py"
            if not dataset_module_path.exists():
                continue
            module = self._load_dataset_module(dataset_dir.name, dataset_module_path)
            definition = getattr(module, "DATASET_DEFINITION", None)
            list_scenes = getattr(module, "list_dataset_scenes", None)
            build_config = getattr(module, "build_dataset_config", None)
            adapter_class = getattr(module, "ADAPTER_CLASS", None)
            if (
                not isinstance(definition, dict)
                or not callable(list_scenes)
                or not callable(build_config)
                or adapter_class is None
                or not isinstance(adapter_class, type)
                or not issubclass(adapter_class, DatasetAdapter)
            ):
                continue
            dataset_id = str(definition.get("id") or "").strip()
            name = str(definition.get("name") or dataset_id).strip()
            if not dataset_id:
                continue
            definitions[dataset_id] = DatasetDefinitionSpec(
                dataset_id=dataset_id,
                name=name,
                class_names=list(definition.get("class_names") or []),
                default_scene_id=definition.get("default_scene_id"),
                list_scenes=list_scenes,
                build_config=build_config,
                adapter_class=adapter_class,
            )
        return dict(sorted(definitions.items(), key=lambda item: item[1].name.lower()))

    def _load_dataset_module(self, dataset_id: str, dataset_module_path: Path) -> ModuleType:
        module_name = f"radar_annotation_dataset_{dataset_id}"
        spec = importlib.util.spec_from_file_location(module_name, dataset_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load dataset plugin: {dataset_module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _catalog_entry_from_spec(self, spec: DatasetDefinitionSpec) -> DatasetCatalogEntry:
        raw_scenes = spec.list_scenes() or []
        scenes = [
            item if isinstance(item, DatasetSceneOption) else DatasetSceneOption.model_validate(item)
            for item in raw_scenes
        ]
        return DatasetCatalogEntry(
            id=spec.dataset_id,
            name=spec.name,
            class_names=spec.class_names,
            default_scene_id=spec.default_scene_id,
            scenes=scenes,
        )

    def _choose_default_scene(self, dataset: DatasetCatalogEntry) -> str | None:
        if dataset.default_scene_id:
            preferred = next((scene for scene in dataset.scenes if scene.id == dataset.default_scene_id and scene.available), None)
            if preferred:
                return preferred.id
        first_available = next((scene for scene in dataset.scenes if scene.available), None)
        if first_available:
            return first_available.id
        return dataset.default_scene_id or (dataset.scenes[0].id if dataset.scenes else None)

    def list_sequences(self) -> list[SequenceSummary]:
        return [self.get_summary(sequence_id) for sequence_id in self._sequences]

    def get(self, sequence_id: str) -> SequenceState:
        return self._sequences[sequence_id]

    def get_summary(self, sequence_id: str) -> SequenceSummary:
        state = self.get(sequence_id)
        metadata = state.adapter.get_metadata()
        return SequenceSummary(
            id=sequence_id,
            name=metadata.sequence_name,
            class_names=metadata.class_names,
            total_frames=len(state.adapter),
            fps=metadata.fps,
        )

    def get_timeline(self, sequence_id: str) -> SequenceTimelineResponse:
        state = self.get(sequence_id)
        frames = [TimelineFrame.model_validate(state.adapter.get_frame_summary(idx)) for idx in range(len(state.adapter))]
        return SequenceTimelineResponse(
            sequence=self.get_summary(sequence_id),
            frames=frames,
        )
