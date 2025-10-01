import external_hyrax_example


def test_version():
    """Check to see that we can get the package version"""
    assert external_hyrax_example._version is not None
