from backend.app.execution.executor import ProcedureExecutor


class FakeCursor:
    description = [("tool_id",), ("tool_name",), ("adoption_count",)]

    def __init__(self):
        self.query = None
        self.parameters = None

    def execute(self, query, parameters):
        self.query = query
        self.parameters = parameters

    def fetchall(self):
        return [(1, "ChatGPT", 42)]

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_postgres_executor_calls_approved_function_with_bound_values():
    connection = FakeConnection()
    executor = ProcedureExecutor(
        connection=connection,
        database_url="postgresql://db-user:secret@db.example/neondb?sslmode=require",
    )

    rows = executor.execute_procedure("get_tool_ranking", {"time_period": "last_month", "limit": 3})

    assert connection.cursor_instance.query == "SELECT * FROM get_tool_ranking(%s, %s)"
    assert connection.cursor_instance.parameters == (30, 3)
    assert rows == [{"tool_id": 1, "tool_name": "ChatGPT", "adoption_count": 42}]


def test_placeholder_database_url_keeps_sqlite_fallback():
    executor = ProcedureExecutor(database_url="postgresql://USER:PASSWORD@HOST/neondb?sslmode=require")

    assert executor.backend == "sqlite"
