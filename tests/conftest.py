"""Pytest configuration for jax-bandflux tests.

Enables float64 precision for all tests. Library code no longer
sets this globally — users must opt in.
"""
import jax
jax.config.update("jax_enable_x64", True)
