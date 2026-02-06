# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Zodiacal is a blind astrometry library written in Rust.

## External References

- `external/starfield/` — Astronomical data reduction toolkit (star catalogs, coordinate systems, star finding). Use as the primary dependency for astronomical types and primitives.
- `external/astrometry.net/` — Reference implementation of blind astrometry (C/Python). Use for algorithmic reference.

## Starfield Integration

- Prefer using types from the `starfield` crate (coordinates, catalogs, time, etc.) rather than reinventing them.
- When starfield types or APIs are missing functionality or could be improved for zodiacal's use case, open an issue on the starfield repo (`gh issue create -R OrbitalCommons/starfield`) describing the improvement.
- Reference `external/starfield/CLAUDE.md` for starfield's conventions and architecture.
