def test_package_imports():
    import touch  # noqa: F401
    import touch.data  # noqa: F401
    import touch.eval  # noqa: F401
    import touch.models  # noqa: F401
