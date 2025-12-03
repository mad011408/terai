"""
Tool safeguards for controlling tool execution permissions.
"""

from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class PermissionLevel(Enum):
    """Permission levels for tool access."""
    NONE = 0
    READ_ONLY = 1
    LIMITED = 2
    STANDARD = 3
    ELEVATED = 4
    ADMIN = 5


class RiskLevel(Enum):
    """Risk levels for tool operations."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolPermission:
    """Permission configuration for a tool."""
    tool_name: str
    allowed: bool
    permission_level: PermissionLevel
    risk_level: RiskLevel
    requires_confirmation: bool
    rate_limit: Optional[int] = None  # Max calls per minute
    allowed_parameters: Optional[Dict[str, Any]] = None
    blocked_parameters: Optional[Dict[str, Any]] = None
    conditions: List[str] = field(default_factory=list)


@dataclass
class ExecutionRequest:
    """Request to execute a tool."""
    tool_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    requester: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionDecision:
    """Decision on whether to allow tool execution."""
    allowed: bool
    reason: str
    requires_confirmation: bool
    modified_parameters: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class ToolSafeguard:
    """
    Controls tool execution with permission checks and safety validation.
    """

    def __init__(self, default_permission_level: PermissionLevel = PermissionLevel.STANDARD):
        self.default_permission_level = default_permission_level
        self.permissions: Dict[str, ToolPermission] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.rate_limits: Dict[str, List[datetime]] = {}
        self._load_default_permissions()

    def _load_default_permissions(self) -> None:
        """Load default tool permissions."""
        default_permissions = [
            # Safe tools
            ToolPermission(
                tool_name="web_search",
                allowed=True,
                permission_level=PermissionLevel.READ_ONLY,
                risk_level=RiskLevel.SAFE,
                requires_confirmation=False,
                rate_limit=60
            ),
            ToolPermission(
                tool_name="file_reader",
                allowed=True,
                permission_level=PermissionLevel.READ_ONLY,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False
            ),
            # Medium risk tools
            ToolPermission(
                tool_name="file_writer",
                allowed=True,
                permission_level=PermissionLevel.STANDARD,
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=True
            ),
            ToolPermission(
                tool_name="api_caller",
                allowed=True,
                permission_level=PermissionLevel.STANDARD,
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=False,
                rate_limit=30
            ),
            # High risk tools
            ToolPermission(
                tool_name="terminal_executor",
                allowed=True,
                permission_level=PermissionLevel.ELEVATED,
                risk_level=RiskLevel.HIGH,
                requires_confirmation=True,
                blocked_parameters={"command": ["rm -rf", "format", "mkfs"]}
            ),
            ToolPermission(
                tool_name="code_executor",
                allowed=True,
                permission_level=PermissionLevel.ELEVATED,
                risk_level=RiskLevel.HIGH,
                requires_confirmation=True
            ),
            # Critical tools
            ToolPermission(
                tool_name="database_query",
                allowed=True,
                permission_level=PermissionLevel.ADMIN,
                risk_level=RiskLevel.CRITICAL,
                requires_confirmation=True,
                conditions=["read_only_mode"]
            ),
        ]

        for perm in default_permissions:
            self.permissions[perm.tool_name] = perm

    def set_permission(self, permission: ToolPermission) -> None:
        """Set permission for a tool."""
        self.permissions[permission.tool_name] = permission

    def get_permission(self, tool_name: str) -> Optional[ToolPermission]:
        """Get permission for a tool."""
        return self.permissions.get(tool_name)

    def check_execution(self, request: ExecutionRequest,
                       user_permission_level: PermissionLevel = PermissionLevel.STANDARD) -> ExecutionDecision:
        """
        Check if a tool execution should be allowed.

        Args:
            request: The execution request
            user_permission_level: The requester's permission level

        Returns:
            ExecutionDecision with the verdict
        """
        tool_name = request.tool_name
        permission = self.permissions.get(tool_name)

        # Unknown tool - use default settings
        if not permission:
            return ExecutionDecision(
                allowed=self.default_permission_level.value >= PermissionLevel.STANDARD.value,
                reason="Unknown tool - using default permissions",
                requires_confirmation=True,
                warnings=["Tool not in permission registry"]
            )

        # Check if tool is allowed at all
        if not permission.allowed:
            return ExecutionDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not allowed",
                requires_confirmation=False
            )

        # Check permission level
        if user_permission_level.value < permission.permission_level.value:
            return ExecutionDecision(
                allowed=False,
                reason=f"Insufficient permissions. Required: {permission.permission_level.name}, Have: {user_permission_level.name}",
                requires_confirmation=False
            )

        # Check rate limit
        if permission.rate_limit:
            if not self._check_rate_limit(tool_name, permission.rate_limit):
                return ExecutionDecision(
                    allowed=False,
                    reason=f"Rate limit exceeded for '{tool_name}'",
                    requires_confirmation=False
                )

        # Check blocked parameters
        warnings = []
        modified_params = request.parameters.copy()

        if permission.blocked_parameters:
            for param, blocked_values in permission.blocked_parameters.items():
                if param in request.parameters:
                    param_value = str(request.parameters[param]).lower()
                    for blocked in blocked_values:
                        if blocked.lower() in param_value:
                            return ExecutionDecision(
                                allowed=False,
                                reason=f"Parameter '{param}' contains blocked value",
                                requires_confirmation=False
                            )

        # Check allowed parameters
        if permission.allowed_parameters:
            for param, allowed_values in permission.allowed_parameters.items():
                if param in request.parameters:
                    if request.parameters[param] not in allowed_values:
                        warnings.append(f"Parameter '{param}' has non-standard value")

        # Check conditions
        if permission.conditions:
            for condition in permission.conditions:
                if not self._check_condition(condition, request.context):
                    return ExecutionDecision(
                        allowed=False,
                        reason=f"Condition not met: {condition}",
                        requires_confirmation=False
                    )

        # Record execution attempt
        self._record_execution(request, True)

        return ExecutionDecision(
            allowed=True,
            reason="All checks passed",
            requires_confirmation=permission.requires_confirmation,
            modified_parameters=modified_params if modified_params != request.parameters else None,
            warnings=warnings
        )

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if rate limit allows execution."""
        now = datetime.now()
        window = timedelta(minutes=1)

        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = []

        # Clean old entries
        self.rate_limits[tool_name] = [
            t for t in self.rate_limits[tool_name]
            if now - t < window
        ]

        if len(self.rate_limits[tool_name]) >= limit:
            return False

        self.rate_limits[tool_name].append(now)
        return True

    def _check_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Check if a condition is met."""
        # Simple condition checking
        if condition == "read_only_mode":
            return context.get("read_only", True)
        elif condition == "sandbox_mode":
            return context.get("sandbox", True)
        elif condition == "authenticated":
            return context.get("authenticated", False)

        return True

    def _record_execution(self, request: ExecutionRequest, allowed: bool) -> None:
        """Record an execution attempt."""
        self.execution_history.append({
            "tool_name": request.tool_name,
            "requester": request.requester,
            "allowed": allowed,
            "timestamp": request.timestamp.isoformat(),
            "parameters_count": len(request.parameters)
        })

        # Keep only last 1000 entries
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get safeguard statistics."""
        total = len(self.execution_history)
        allowed = sum(1 for e in self.execution_history if e["allowed"])

        tool_usage = {}
        for entry in self.execution_history:
            tool = entry["tool_name"]
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

        return {
            "total_requests": total,
            "allowed": allowed,
            "blocked": total - allowed,
            "allow_rate": allowed / total if total > 0 else 0,
            "tool_usage": tool_usage,
            "registered_tools": len(self.permissions)
        }


class ConfirmationHandler:
    """
    Handles user confirmations for dangerous operations.
    """

    def __init__(self):
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}
        self.confirmed_patterns: Set[str] = set()

    def request_confirmation(self, request: ExecutionRequest,
                            decision: ExecutionDecision) -> str:
        """
        Request user confirmation for an execution.

        Returns:
            Confirmation ID
        """
        import uuid
        confirmation_id = str(uuid.uuid4())[:8]

        self.pending_confirmations[confirmation_id] = {
            "request": request,
            "decision": decision,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=5)
        }

        return confirmation_id

    def confirm(self, confirmation_id: str, confirmed: bool,
               remember: bool = False) -> bool:
        """
        Process a confirmation response.

        Args:
            confirmation_id: The confirmation ID
            confirmed: Whether the user confirmed
            remember: Whether to remember this pattern

        Returns:
            Whether the confirmation was processed
        """
        if confirmation_id not in self.pending_confirmations:
            return False

        pending = self.pending_confirmations[confirmation_id]

        # Check expiration
        if datetime.now() > pending["expires_at"]:
            del self.pending_confirmations[confirmation_id]
            return False

        if confirmed and remember:
            # Remember this tool+parameter pattern
            request = pending["request"]
            pattern = f"{request.tool_name}:{hash(frozenset(request.parameters.items()))}"
            self.confirmed_patterns.add(pattern)

        del self.pending_confirmations[confirmation_id]
        return True

    def is_pre_confirmed(self, request: ExecutionRequest) -> bool:
        """Check if a request matches a pre-confirmed pattern."""
        pattern = f"{request.tool_name}:{hash(frozenset(request.parameters.items()))}"
        return pattern in self.confirmed_patterns

    def get_confirmation_message(self, request: ExecutionRequest,
                                decision: ExecutionDecision) -> str:
        """Generate a user-friendly confirmation message."""
        permission = None
        # Would get from safeguard

        message = f"⚠️ Confirmation Required\n\n"
        message += f"Tool: {request.tool_name}\n"

        if decision.warnings:
            message += f"Warnings: {', '.join(decision.warnings)}\n"

        message += f"\nParameters:\n"
        for key, value in request.parameters.items():
            # Truncate long values
            display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            message += f"  - {key}: {display_value}\n"

        message += f"\nDo you want to proceed? [y/N]"
        return message


class SandboxedExecution:
    """
    Provides sandboxed execution environment for tools.
    """

    def __init__(self, safeguard: ToolSafeguard):
        self.safeguard = safeguard
        self.sandbox_config = {
            "file_system_root": "/tmp/sandbox",
            "network_allowed": False,
            "max_memory_mb": 512,
            "max_cpu_seconds": 30,
            "allowed_imports": ["os.path", "json", "re", "datetime"],
            "blocked_imports": ["subprocess", "socket", "shutil", "os.system"],
        }

    def execute_in_sandbox(self, tool: Any, request: ExecutionRequest) -> Dict[str, Any]:
        """
        Execute a tool in a sandboxed environment.

        Args:
            tool: The tool to execute
            request: Execution request

        Returns:
            Execution result
        """
        # Apply sandbox restrictions
        sandboxed_params = self._apply_sandbox_restrictions(request.parameters)

        # Create sandboxed context
        sandboxed_context = {
            "sandbox": True,
            "read_only": True,
            **request.context
        }

        # Check permissions with sandbox context
        decision = self.safeguard.check_execution(
            ExecutionRequest(
                tool_name=request.tool_name,
                parameters=sandboxed_params,
                context=sandboxed_context,
                requester=request.requester
            ),
            user_permission_level=PermissionLevel.STANDARD
        )

        if not decision.allowed:
            return {
                "success": False,
                "error": f"Sandbox execution blocked: {decision.reason}"
            }

        # Execute (actual sandboxing would be more complex)
        try:
            # In real implementation, would use actual sandboxing
            result = {"success": True, "result": "Sandboxed execution placeholder"}
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_sandbox_restrictions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sandbox restrictions to parameters."""
        restricted = parameters.copy()

        # Restrict file paths to sandbox root
        for key, value in restricted.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                # Assume it's a path
                if not value.startswith(self.sandbox_config["file_system_root"]):
                    restricted[key] = f"{self.sandbox_config['file_system_root']}/{value.lstrip('/')}"

        return restricted
