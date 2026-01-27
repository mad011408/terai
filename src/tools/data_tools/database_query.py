"""
Database query tool for data operations.
Supports multiple database types.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    connection_string: Optional[str] = None


@dataclass
class QueryResult:
    """Result of a database query."""
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time: float
    error: Optional[str] = None


class DatabaseQueryTool(BaseTool):
    """
    Tool for executing database queries.
    Supports read-only queries for safety.
    """

    def __init__(self, config: DatabaseConfig):
        tool_config = ToolConfig(
            name="database_query",
            description="Execute read-only database queries to retrieve data.",
            category=ToolCategory.DATA,
            timeout=60.0,
            retry_attempts=2
        )
        super().__init__(tool_config)
        self.db_config = config
        self.connection = None
        self._connected = False

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description="SQL query to execute (SELECT only)",
                required=True
            ),
            ToolParameter(
                name="parameters",
                param_type="object",
                description="Query parameters for parameterized queries",
                required=False,
                default={}
            ),
            ToolParameter(
                name="limit",
                param_type="number",
                description="Maximum number of rows to return",
                required=False,
                default=100,
                min_value=1,
                max_value=10000
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Query timeout in seconds",
                required=False,
                default=30.0,
                min_value=1,
                max_value=300
            )
        ]

    def _validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate query is read-only."""
        query_upper = query.strip().upper()

        # Check for dangerous operations
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
        ]

        for keyword in dangerous_keywords:
            if query_upper.startswith(keyword) or f" {keyword} " in f" {query_upper} ":
                return False, f"Query contains forbidden keyword: {keyword}"

        # Must be a SELECT query
        if not query_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed"

        return True, None

    async def _execute(self, query: str, parameters: Dict = None,
                      limit: int = 100, timeout: float = 30.0) -> QueryResult:
        """Execute the database query."""
        # Validate query
        is_valid, error = self._validate_query(query)
        if not is_valid:
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                columns=[],
                execution_time=0,
                error=error
            )

        # Add LIMIT if not present
        if "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"

        try:
            if self.db_config.db_type == DatabaseType.SQLITE:
                return await self._execute_sqlite(query, parameters, timeout)
            elif self.db_config.db_type == DatabaseType.POSTGRESQL:
                return await self._execute_postgresql(query, parameters, timeout)
            elif self.db_config.db_type == DatabaseType.MYSQL:
                return await self._execute_mysql(query, parameters, timeout)
            elif self.db_config.db_type == DatabaseType.MONGODB:
                return await self._execute_mongodb(query, parameters, timeout)
            else:
                return QueryResult(
                    success=False,
                    rows=[],
                    row_count=0,
                    columns=[],
                    execution_time=0,
                    error=f"Unsupported database type: {self.db_config.db_type}"
                )
        except Exception as e:
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                columns=[],
                execution_time=0,
                error=str(e)
            )

    async def _execute_sqlite(self, query: str, parameters: Dict,
                             timeout: float) -> QueryResult:
        """Execute SQLite query."""
        import aiosqlite
        import time

        start_time = time.time()

        async with aiosqlite.connect(self.db_config.database) as db:
            db.row_factory = aiosqlite.Row

            if parameters:
                cursor = await db.execute(query, parameters)
            else:
                cursor = await db.execute(query)

            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []

            result_rows = [dict(row) for row in rows]

            return QueryResult(
                success=True,
                rows=result_rows,
                row_count=len(result_rows),
                columns=columns,
                execution_time=time.time() - start_time
            )

    async def _execute_postgresql(self, query: str, parameters: Dict,
                                  timeout: float) -> QueryResult:
        """Execute PostgreSQL query."""
        import asyncpg
        import time

        start_time = time.time()

        conn = await asyncpg.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.username,
            password=self.db_config.password,
            timeout=timeout
        )

        try:
            if parameters:
                rows = await conn.fetch(query, *parameters.values())
            else:
                rows = await conn.fetch(query)

            columns = list(rows[0].keys()) if rows else []
            result_rows = [dict(row) for row in rows]

            return QueryResult(
                success=True,
                rows=result_rows,
                row_count=len(result_rows),
                columns=columns,
                execution_time=time.time() - start_time
            )
        finally:
            await conn.close()

    async def _execute_mysql(self, query: str, parameters: Dict,
                            timeout: float) -> QueryResult:
        """Execute MySQL query."""
        import aiomysql
        import time

        start_time = time.time()

        conn = await aiomysql.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            db=self.db_config.database,
            user=self.db_config.username,
            password=self.db_config.password,
            connect_timeout=timeout
        )

        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if parameters:
                    await cursor.execute(query, tuple(parameters.values()))
                else:
                    await cursor.execute(query)

                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []

                return QueryResult(
                    success=True,
                    rows=list(rows),
                    row_count=len(rows),
                    columns=columns,
                    execution_time=time.time() - start_time
                )
        finally:
            conn.close()

    async def _execute_mongodb(self, query: str, parameters: Dict,
                              timeout: float) -> QueryResult:
        """Execute MongoDB query (using JSON query format)."""
        from motor.motor_asyncio import AsyncIOMotorClient
        import time
        import json

        start_time = time.time()

        # Parse query as JSON
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                columns=[],
                execution_time=0,
                error="MongoDB query must be valid JSON"
            )

        client = AsyncIOMotorClient(
            self.db_config.host,
            self.db_config.port,
            serverSelectionTimeoutMS=int(timeout * 1000)
        )

        try:
            db = client[self.db_config.database]
            collection_name = query_dict.get("collection", "default")
            collection = db[collection_name]

            filter_query = query_dict.get("filter", {})
            projection = query_dict.get("projection", None)
            limit = query_dict.get("limit", 100)

            cursor = collection.find(filter_query, projection).limit(limit)
            rows = await cursor.to_list(length=limit)

            # Convert ObjectId to string
            for row in rows:
                if "_id" in row:
                    row["_id"] = str(row["_id"])

            columns = list(rows[0].keys()) if rows else []

            return QueryResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                columns=columns,
                execution_time=time.time() - start_time
            )
        finally:
            client.close()

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            result = await self._execute("SELECT 1", {}, 100, 5.0)
            return result.success
        except Exception:
            return False

    def get_schema_query(self) -> str:
        """Get query to retrieve database schema."""
        if self.db_config.db_type == DatabaseType.SQLITE:
            return "SELECT name, sql FROM sqlite_master WHERE type='table'"
        elif self.db_config.db_type == DatabaseType.POSTGRESQL:
            return """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
            """
        elif self.db_config.db_type == DatabaseType.MYSQL:
            return """
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
            """
        else:
            return ""


class RedisQueryTool(BaseTool):
    """
    Tool for querying Redis.
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 password: Optional[str] = None, db: int = 0):
        config = ToolConfig(
            name="redis_query",
            description="Query data from Redis.",
            category=ToolCategory.DATA,
            timeout=30.0
        )
        super().__init__(config)
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                param_type="string",
                description="Redis command (GET, HGETALL, LRANGE, etc.)",
                required=True,
                enum_values=["GET", "HGETALL", "LRANGE", "SMEMBERS", "ZRANGE", "KEYS", "EXISTS", "TYPE"]
            ),
            ToolParameter(
                name="key",
                param_type="string",
                description="Redis key to query",
                required=True
            ),
            ToolParameter(
                name="args",
                param_type="array",
                description="Additional arguments for the command",
                required=False,
                default=[]
            )
        ]

    async def _execute(self, command: str, key: str, args: List = None) -> Any:
        """Execute Redis command."""
        import aioredis

        if not self.client:
            self.client = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password
            )

        command_upper = command.upper()
        args = args or []

        if command_upper == "GET":
            return await self.client.get(key)
        elif command_upper == "HGETALL":
            return await self.client.hgetall(key)
        elif command_upper == "LRANGE":
            start = args[0] if len(args) > 0 else 0
            stop = args[1] if len(args) > 1 else -1
            return await self.client.lrange(key, start, stop)
        elif command_upper == "SMEMBERS":
            return await self.client.smembers(key)
        elif command_upper == "ZRANGE":
            start = args[0] if len(args) > 0 else 0
            stop = args[1] if len(args) > 1 else -1
            return await self.client.zrange(key, start, stop)
        elif command_upper == "KEYS":
            return await self.client.keys(key)
        elif command_upper == "EXISTS":
            return await self.client.exists(key)
        elif command_upper == "TYPE":
            return await self.client.type(key)
        else:
            raise ValueError(f"Unsupported command: {command}")

    async def close(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.client = None
