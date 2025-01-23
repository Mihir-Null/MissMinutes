from langchain_core.tools import tool
from typing import Dict, Any
from datetime import datetime
from dida365 import Dida365Client, TaskCreate, TaskUpdate, TaskPriority, ProjectCreate, ProjectUpdate, ViewMode, ProjectKind, ApiError, AuthenticationError

# The docstring in tool definition is very important for the agent to work as the agent will use it to determine which tool to use
# when you add your  own tools, please write the doc string as detailed and comprehensive as possible.

# Single client instance and inbox id cache
_client = None
_inbox_id = None

def get_client():
    """Get or initialize the TickTick client singleton"""
    global _client
    if _client is None:
        _client = Dida365Client()
    return _client

async def ensure_client_initialized():
    """Ensure client is authenticated"""
    client = get_client()
    if not client.auth.token: 
        await client.authenticate()

async def get_inbox_id():
    """Get the inbox project ID, fetching and caching if necessary"""
    global _inbox_id
    if _inbox_id is None:
        client = get_client()
        await ensure_client_initialized()
        # Create a task without project_id to get the inbox ID
        temp_task = await client.create_task(TaskCreate(title="Temporary task for inbox ID", project_id=""))
        _inbox_id = temp_task.project_id
        # Clean up the temporary task
        await client.delete_task(_inbox_id, temp_task.id)
    return _inbox_id

def format_tool_error(error: Exception, tool_name: str) -> str:
    """Format API errors for model consumption"""
    # Let authentication errors raise to user
    if isinstance(error, AuthenticationError):
        raise error
        
    # Only handle API errors for model
    if isinstance(error, ApiError):
        print(f"API Error: {str(error)}")
        return f"""<tool_error>{str(error)}</tool_error>"""
        
    # All other errors should raise to user
    raise error

@tool
async def add_task(
    title: str,
    project_id: str = "",
    content: str = "",
    priority: int = 0,
    start_date: str = None,
    due_date: str = None,
    is_all_day: bool = False,
    items: list = None
) -> str:
    """Add a new task to TickTick.
    
    Args:
        title: The title of the task
        project_id: Project ID to add task to. Use empty string "" for inbox
        content: Detailed content/notes for the task
        priority: Task priority (none=0, low=1, medium=3, high=5)
        start_date: Start date/time in ISO format (e.g. "2024-03-20T15:00:00Z")
        due_date: Due date/time in ISO format
        is_all_day: Whether task is all-day
        items: List of checklist items/subtasks
            Checklist item is a dictionary with the following keys:
            checklist_item = {
                "title": "Subtask",           # Required: Item title
                "status": 0,                  # Optional: 0=normal, 1=completed
                "start_date": datetime.now(), # Optional: Start time
                "is_all_day": False,         # Optional: All-day item
                "time_zone": "UTC"           # Optional: Time zone
            }
    Returns:
        A confirmation message with the task ID and project ID
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        # Build task dict
        task_dict = {
            "title": title,
            "project_id": project_id,
            "content": content,
            "priority": priority,
            "is_all_day": is_all_day
        }

        # Convert priority if provided
        priority_map = {
            0: TaskPriority.NONE, 
            1: TaskPriority.LOW,
            3: TaskPriority.MEDIUM, 
            5: TaskPriority.HIGH
        }
        task_dict["priority"] = priority_map.get(priority, TaskPriority.NONE)

        # Parse dates if provided
        if start_date:
            task_dict["start_date"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if due_date:
            task_dict["due_date"] = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
        
        # Add items if provided
        if items:
            task_dict["items"] = items

        # Create task
        task = await client.create_task(TaskCreate(**task_dict))
        
        # If project_id was empty, cache the inbox ID
        global _inbox_id
        if not project_id and _inbox_id is None:
            _inbox_id = task.project_id
        
        return f"Created task '{task.title}' with ID: {task.id} in project (project_id): {task.project_id}"
    except Exception as e:
        return format_tool_error(e, "add_task")

@tool
async def complete_task(project_id: str, task_id: str) -> str:
    """Mark a task as completed in TickTick.
    
    Args:
        project_id: The ID of the project containing the task
        task_id: The ID of the task to complete
        
    Returns:
        A confirmation message
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        if not project_id:
            project_id = await get_inbox_id()
        await client.complete_task(project_id, task_id)
        return f"Marked task {task_id} as complete"
    except Exception as e:
        return format_tool_error(e, "complete_task")

@tool
async def get_task(project_id: str, task_id: str) -> str:
    """Get details of a specific task.
    
    Args:
        project_id: The ID of the project containing the task
        task_id: The ID of the task to retrieve
        
    Returns:
        Task details including title, content, priority, dates, etc.
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        if not project_id:
            project_id = await get_inbox_id()
            
        task = await client.get_task(project_id, task_id)
        return (
            f"Task '{task.title}'\n"
            f"ID: {task.id}\n"
            f"Project ID: {task.project_id}\n"
            f"Content: {task.content or 'No content'}\n"
            f"Priority: {task.priority}\n"
            f"Start date: {task.start_date or 'Not set'}\n"
            f"Due date: {task.due_date or 'Not set'}\n"
            f"Status: {task.status}\n"
            f"Subtasks: {len(task.items) if task.items else 0}"
        )
    except Exception as e:
        return format_tool_error(e, "get_task")

@tool
async def update_task(
    task_id: str,
    project_id: str,
    title: str = None,
    content: str = None,
    priority: int = None,
    start_date: str = None,
    due_date: str = None,
    is_all_day: bool = None,
    items: list = None
) -> str:
    """Update an existing task's properties.
    
    Args:
        task_id: The ID of the task to update
        project_id: The ID of the project containing the task
        title: New task title (optional)
        content: New task content/notes (optional)
        priority: New priority level (0=none, 1=low, 3=medium, 5=high) (optional)
        start_date: New start date in ISO format (optional)
        due_date: New due date in ISO format (optional)
        is_all_day: Whether task is all-day (optional)
        items: New list of checklist items (optional)
        
    Returns:
        Confirmation of the update with task details
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        # Build update dict with only provided values
        update_dict = {
            "id": task_id,
            "project_id": project_id
        }
        
        if title is not None:
            update_dict["title"] = title
        if content is not None:
            update_dict["content"] = content
        if priority is not None:
            priority_map = {
                0: TaskPriority.NONE,
                1: TaskPriority.LOW,
                3: TaskPriority.MEDIUM,
                5: TaskPriority.HIGH
            }
            update_dict["priority"] = priority_map.get(priority, TaskPriority.NONE)
        if start_date is not None:
            update_dict["start_date"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if due_date is not None:
            update_dict["due_date"] = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
        if is_all_day is not None:
            update_dict["is_all_day"] = is_all_day
        if items is not None:
            update_dict["items"] = items

        task = await client.update_task(TaskUpdate(**update_dict))
        return f"Updated task '{task.title}' with ID: {task.id} in project (project_id): {task.project_id}"
    except Exception as e:
        return format_tool_error(e, "update_task")

@tool
async def delete_task(project_id: str, task_id: str) -> str:
    """Delete a task permanently.
    
    Args:
        project_id: The ID of the project containing the task. Use empty string "" for inbox
        task_id: The ID of the task to delete
        
    Returns:
        Confirmation of deletion
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        # If project_id is empty, use inbox_id
        if not project_id:
            project_id = await get_inbox_id()
        
        await client.delete_task(project_id, task_id)
        return f"Deleted task {task_id} from project (project_id): {project_id}"
    except Exception as e:
        return format_tool_error(e, "delete_task")

@tool
async def get_project_tasks(project_id: str) -> str:
    """Get all tasks'title and id in a project. 
    
    Args:
        project_id: The ID of the project to get tasks from
        
    Returns:
        List of tasks in the project
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        project_data = await client.get_project_with_data(project_id)
        tasks = project_data.tasks
        
        if not tasks:
            return f"No tasks found in project (project_id): {project_id}"
        
        task_list = [f"- {task.title} (ID: {task.id})" for task in tasks]
        return (
            f"Found {len(tasks)} tasks in project (project_id): {project_id}\n"
            + "\n".join(task_list)
        )
    except Exception as e:
        return format_tool_error(e, "get_project_tasks")

@tool
async def get_project_tasks_detailed_with_data(project_id: str) -> str:
    """Get detailed information about all tasks in a project. 
    It is the same as get_project_tasks and do get_task iteratively, you get the entire project data in one call.
    Args:
        project_id: The ID of the project to get tasks from
        
    Returns:
        Detailed list of tasks including title, ID, priority, dates, content, and subtasks
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        project_data = await client.get_project_with_data(project_id)
        tasks = project_data.tasks
        
        if not tasks:
            return f"No tasks found in project (project_id): {project_id}"
        
        priority_map = {
            0: "None",
            1: "Low",
            3: "Medium",
            5: "High"
        }
        
        task_details = []
        for task in tasks:
            # Format dates nicely if they exist
            start_date = task.start_date.strftime("%Y-%m-%d %H:%M") if task.start_date else "Not set"
            due_date = task.due_date.strftime("%Y-%m-%d %H:%M") if task.due_date else "Not set"
            
            # Build subtasks list if any
            subtasks = []
            if task.items:
                for item in task.items:
                    status = "✓" if item.status == 1 else "○"
                    subtasks.append(f"  {status} {item.title}")
            
            # Build task details
            details = [
                f"Task: {task.title}",
                f"ID: {task.id}",
                f"Priority: {priority_map.get(task.priority, 'Unknown')}",
                f"Status: {'Completed' if task.status == 2 else 'Active'}",
                f"Start Date: {start_date}",
                f"Due Date: {due_date}",
                f"Content: {task.content or 'No content'}"
            ]
            
            if subtasks:
                details.append("Subtasks:")
                details.extend(subtasks)
                
            task_details.append("\n".join(details))
        
        return (
            f"Found {len(tasks)} tasks in project (project_id): {project_id}\n\n"
            + "\n\n".join(task_details)
        )
    except Exception as e:
        return format_tool_error(e, "get_project_tasks_detailed_with_data")

@tool
async def create_project(
    name: str,
    color: str = None,
    view_mode: str = None,
    kind: str = None
) -> str:
    """Create a new project in TickTick.
    
    Args:
        name: Project name
        color: Hex color code (e.g. "#FF0000") (optional)
        view_mode: View mode (LIST, KANBAN, TIMELINE) (optional)
        kind: Project kind (TASK, NOTE) (optional)
    
    Returns:
        A confirmation message with the project ID
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        # Build project dict
        project_dict = {
            "name": name
        }
        
        if color:
            project_dict["color"] = color
        if view_mode:
            project_dict["view_mode"] = ViewMode(view_mode)
        if kind:
            project_dict["kind"] = ProjectKind(kind)

        project = await client.create_project(ProjectCreate(**project_dict))
        return f"Created project '{project.name}' with ID: {project.id}"
    except Exception as e:
        return format_tool_error(e, "create_project")

@tool
async def get_project(project_id: str) -> str:
    """Get details of a specific project.
    
    Args:
        project_id: The ID of the project to retrieve
        
    Returns:
        Project details including name, color, view mode, etc.
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        project = await client.get_project(project_id)
        return (
            f"Project '{project.name}'\n"
            f"ID: {project.id}\n"
            f"Color: {project.color or 'Default'}\n"
            f"View Mode: {project.view_mode}\n"
            f"Kind: {project.kind}\n"
            f"Status: {'Closed' if project.closed else 'Open'}"
        )
    except Exception as e:
        return format_tool_error(e, "get_project")

@tool
async def get_all_projects() -> str:
    """Get a list of all projects.
    
    Returns:
        List of all projects with their IDs and names
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        projects = await client.get_projects()
        
        if not projects:
            return "No projects found"
        
        project_list = [f"- {project.name} (ID: {project.id})" for project in projects]
        return (
            f"Found {len(projects)} projects:\n"
            + "\n".join(project_list)
        )
    except Exception as e:
        return format_tool_error(e, "get_all_projects")

@tool
async def update_project(
    project_id: str,
    name: str = None,
    color: str = None,
    view_mode: str = None,
    kind: str = None
) -> str:
    """Update an existing project's properties.
    
    Args:
        project_id: The ID of the project to update
        name: New project name (optional)
        color: New hex color code (optional)
        view_mode: New view mode (LIST, KANBAN, TIMELINE) (optional)
        kind: New project kind (TASK, NOTE) (optional)
        
    Returns:
        Confirmation of the update with project details
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        # Build update dict with only provided values
        update_dict = {
            "id": project_id
        }
        
        if name is not None:
            update_dict["name"] = name
        if color is not None:
            update_dict["color"] = color
        if view_mode is not None:
            update_dict["view_mode"] = ViewMode(view_mode)
        if kind is not None:
            update_dict["kind"] = ProjectKind(kind)

        project = await client.update_project(ProjectUpdate(**update_dict))
        return f"Updated project '{project.name}' with ID: {project.id}"
    except Exception as e:
        return format_tool_error(e, "update_project")

@tool
async def delete_project(project_id: str) -> str:
    """Delete a project permanently.
    
    Args:
        project_id: The ID of the project to delete
        
    Returns:
        Confirmation of deletion
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        await client.delete_project(project_id)
        return f"Deleted project {project_id}"
    except Exception as e:
        return format_tool_error(e, "delete_project")

@tool
async def get_active_projects() -> str:
    """Get a list of all active (non-closed) projects.
    
    Returns:
        List of active projects with their IDs and names
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        projects = await client.get_projects()
        active_projects = [p for p in projects if not p.closed]
        
        if not active_projects:
            return "No active projects found"
        
        project_list = [f"- {project.name} (ID: {project.id})" for project in active_projects]
        return (
            f"Found {len(active_projects)} active projects:\n"
            + "\n".join(project_list)
        )
    except Exception as e:
        return format_tool_error(e, "get_active_projects")

@tool
async def get_closed_projects() -> str:
    """Get a list of all closed (archived) projects.
    
    Returns:
        List of closed projects with their IDs and names
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        projects = await client.get_projects()
        closed_projects = [p for p in projects if p.closed]
        
        if not closed_projects:
            return "No closed projects found"
        
        project_list = [f"- {project.name} (ID: {project.id})" for project in closed_projects]
        return (
            f"Found {len(closed_projects)} closed projects:\n"
            + "\n".join(project_list)
        )
    except Exception as e:
        return format_tool_error(e, "get_closed_projects")

@tool
async def get_inbox_tasks() -> str:
    """Get all tasks from the inbox.
    
    Returns:
        List of tasks in the inbox
        If there's an error, returns an XML-formatted error message.
    """
    try:
        client = get_client()
        await ensure_client_initialized()
        
        inbox_id = await get_inbox_id()
        project_data = await client.get_project_with_data(inbox_id)
        tasks = project_data.tasks
        
        if not tasks:
            return "No tasks found in inbox"
        
        task_list = [f"- {task.title} (ID: {task.id})" for task in tasks]
        return (
            f"Found {len(tasks)} tasks in inbox:\n"
            + "\n".join(task_list)
        )
    except Exception as e:
        return format_tool_error(e, "get_inbox_tasks")
    
    
    
    