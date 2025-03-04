
pytest_test(
    name = "test_external_resources",
    srcs = ["test_external_resources.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic/api/tests:fixtures",
        "//sematic/db/models:external_resource",
        "//sematic/db/models:user",
        "//sematic/db/tests:fixtures",
        "//sematic/plugins:abstract_external_resource",
    ],
)

pytest_test(
    name = "test_resolutions",
    srcs = ["test_resolutions.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic:abstract_future",
        "//sematic/api/tests:fixtures",
        "//sematic/db:queries",
        "//sematic/db/models:resolution",
        "//sematic/db/models:run",
        "//sematic/db/tests:fixtures",
        "//sematic/scheduling:external_job",
        "//sematic/scheduling:kubernetes",
    ],
)

pytest_test(
    name = "test_runs",
    srcs = ["test_runs.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic:calculator",
        "//sematic/api/tests:fixtures",
        "//sematic/db:queries",
        "//sematic/db/tests:fixtures",
        "//sematic/resolvers/tests:fixtures",
        "//sematic/tests:fixtures",
        "//sematic/utils:exceptions",
    ],
)

pytest_test(
    name = "test_notes",
    srcs = ["test_notes.py"],
    pip_deps = [
        "flask",
        "sqlalchemy",
    ],
    deps = [
        "//sematic/api/tests:fixtures",
        "//sematic/db:queries",
        "//sematic/db/models:note",
        "//sematic/db/models:run",
        "//sematic/db/models:user",
        "//sematic/db/tests:fixtures",
    ],
)

pytest_test(
    name = "test_auth",
    srcs = ["test_auth.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic/api/endpoints:auth",
        "//sematic/api/tests:fixtures",
        "//sematic/config:server_settings",
        "//sematic/config:settings",
        "//sematic/db:queries",
        "//sematic/db/models:json_encodable_mixin",
        "//sematic/db/models:user",
    ],
)

pytest_test(
    name = "test_meta",
    srcs = ["test_meta.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic:versions",
        "//sematic/api:app",
        "//sematic/api/endpoints:meta",
        "//sematic/api/tests:fixtures",
        "//sematic/config:server_settings",
    ],
)

pytest_test(
    name = "test_artifacts",
    srcs = ["test_artifacts.py"],
    pip_deps = [
        "flask",
    ],
    deps = [
        "//sematic/api/tests:fixtures",
        "//sematic/db/models:artifact",
        "//sematic/db/tests:fixtures",
        "//sematic/plugins/storage:memory_storage",
        "//sematic/plugins/storage:s3_storage",
        "//sematic/tests:fixtures",
    ],
)
