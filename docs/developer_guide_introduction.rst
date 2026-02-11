============
Introduction
============

This guide is intended for developers who want to understand, modify, or extend the RMS-NAV system. It provides an overview of the system architecture, details on the class hierarchy, and instructions for extending the system with new functionality.

System Architecture
===================

RMS-NAV follows a modular architecture organized around several key components:

1. :class:`~nav.nav_master.nav_master.NavMaster` — The central controller that coordinates the navigation process
2. :class:`~nav.nav_model.nav_model.NavModel` — Generates theoretical models of what should appear in images
3. :class:`~nav.nav_technique.nav_technique.NavTechnique` — Implements algorithms to match models with actual images
4. :class:`~nav.dataset.dataset.DataSet` — Handles image file access and organization
5. :class:`~nav.obs.obs_snapshot.ObsSnapshot` — Manages observation data and coordinate transformations
6. :class:`~nav.annotation.annotation.Annotation` — Creates visual overlays and text annotations

Data Flow
---------

1. The system loads an image through a Dataset implementation
2. An ObsSnapshot is created to represent the observation
3. NavMaster coordinates the creation of models (stars, bodies, rings)
4. Navigation techniques are applied to find the best offset
5. Results are processed to create annotations and overlays
6. Output files are generated (metadata JSON and summary PNG images)
