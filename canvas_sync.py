import os
import yaml
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from datetime import datetime, timedelta
from dotenv import load_dotenv

CONFIG_FILE = "canvas_config.yaml"

def get_config():
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "courses": [
                {
                    "name": "Example Course 101",
                    "url": "/courses/1234567",
                    "ticktick_list_id": "optional_list_id_here",
                }
            ],
            "rules": {
                "priority": {
                    "Exam": 5,
                    "Quiz": 5,
                    "Homework": 3,
                    "Optional": 1
                },
                "tags": {
                    "Homework": "Homework",
                    "Exam": "Exam"
                }
            }
        }
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(default_config, f)
        return default_config
    
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

async def _login_and_get_assignments(base_url, username, password, headless=True):
    """
    Spins up Playwright, logs into UMD CAS, and scrapes the user's dashboard assignments.
    """
    async with async_playwright() as p:
        # We use a persistent context so we don't have to 2FA every time.
        # User will have to run `headless=False` the very first time to pass Duo.
        user_data_dir = os.path.join(os.getcwd(), ".canvas_browser_profile")
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            executable_path="/usr/bin/chromium"
        )

        page = await browser.new_page()

        # Step 1: Navigate to Canvas
        print(f"Navigating to {base_url}...")
        await page.goto(base_url)

        # Step 2: Handle UMD CAS Login if we aren't already authenticated via cookies
        # Yield a bit of time for redirects
        try:
             await page.wait_for_load_state('networkidle', timeout=5000)
        except:
             pass

        if "cas.umd.edu" in page.url or "login" in page.url or "shib.idm.umd.edu" in page.url:
            print("Login required. Attempting to autofill CAS credentials...")
            
            try:
                # Fill CAS Login if elements exist
                username_input = await page.wait_for_selector('input[name="j_username"], input[id="username"]', timeout=5000)
                if username_input:
                    await username_input.fill(username)
                    await page.fill('input[name="j_password"], input[id="password"]', password)
                    await page.click('button[name="_eventId_proceed"], button[type="submit"]')
                    print("Credentials submitted. Waiting for Duo 2FA approval...")
            except Exception as e:
                print("Could not autofill credentials, please enter them manually.")
            
            # Since the user might be doing 2FA manually or the UI might have changed,
            # we simply wait for the URL to change to the Canvas dashboard.
            try:
                # Wait until we hit the actual canvas dashboard
                # We will check the URL continuously until it shows we're logged in
                import time
                start_time = time.time()
                while time.time() - start_time < 120:  # 2 minute timeout
                    if "myelms.umd.edu/?login_success" in page.url or "umd.instructure.com" in page.url:
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                pass
                
        # To be completely safe, we'll just wait for an element that signifies we are logged in.
        try:
             # Wait for the global navigation menu or dashboard header
             await page.wait_for_selector('a#global_nav_profile_link, .ic-Dashboard-header__title', timeout=120000)
             print("Successfully reached Canvas dashboard.")
        except Exception as e:
             print(f"Failed to confirm dashboard. Current URL: {page.url}. Exception: {e}")
             await browser.close()
             return []

        assignments_data = []
        config = get_config()

        # Step 3: Go to the Assignments page for each tracked course
        for course in config.get("courses", []):
            course_url = f"{base_url}{course['url']}/assignments"
            print(f"Scraping assignments for {course['name']} at {course_url}...")
            
            try:
                await page.goto(course_url, wait_until="networkidle")
                
                # Wait for the assignment list to load
                await page.wait_for_selector(".assignment-list", timeout=10000)

                # Parse the assignments
                # Canvas typically wraps assignments in `.ig-row` elements.
                assignment_rows = await page.query_selector_all(".ig-row")
                
                print(f"Found {len(assignment_rows)} potential assignment rows (.ig-row). Parsing them...")
                for row in assignment_rows:
                    title_elem = await row.query_selector("a.ig-title")
                    if not title_elem:
                        print("  -> Row skipped: 'a.ig-title' not found in this row.")
                        continue

                    title = await title_elem.inner_text()
                    href = await title_elem.get_attribute("href")
                    
                    # Get Due Date
                    due_date_str = ""
                    due_date = None
                    
                    # Try current Canvas layout (e.g., "Mar 8 at 11:59pm")
                    date_elem_new = await row.query_selector(".assignment-date-due span[data-html-tooltip-title]")
                    if date_elem_new:
                        due_date_str = await date_elem_new.get_attribute("data-html-tooltip-title")
                        if due_date_str:
                            try:
                                clean_str = due_date_str.replace(" at ", " ").replace("am", " AM").replace("pm", " PM")
                                current_year = datetime.now().year
                                due_date = datetime.strptime(f"{current_year} {clean_str}", "%Y %b %d %I:%M %p")
                            except Exception as e:
                                pass
                    
                    if not due_date:
                        # Fallback to older Canvas layout (ISO string in <time datetime="...">)
                        date_elem_old = await row.query_selector(".ig-details time")
                        if date_elem_old:
                            dt_attr = await date_elem_old.get_attribute("datetime")
                            if dt_attr:
                                try:
                                    due_date = datetime.fromisoformat(dt_attr.replace('Z', '+00:00'))
                                    due_date_str = dt_attr
                                except:
                                    pass

                    # Determine priority/tags based on rules (Global or Course-specific)
                    priority = 3  # default Medium
                    tags = []
                    course_rules = course.get("rules", config.get("rules", {}))
                    
                    for keyword, prio in course_rules.get("priority", {}).items():
                        if keyword.lower() in title.lower():
                            priority = prio
                            break
                    for keyword, tag in course_rules.get("tags", {}).items():
                        if keyword.lower() in title.lower():
                            tags.append(tag)

                    # Determine if it's submitted based on the score text
                    is_submitted = False
                    score_elem = await row.query_selector('.score-display')
                    if score_elem:
                        score_text = await score_elem.inner_text()
                        # If it says 'Score' or has 'pts' without a '-', usually it implies a submission or grade
                        # More robustly, check the tooltip title "No Submission"
                        title_attr = await score_elem.get_attribute('title')
                        if title_attr and "No Submission" not in title_attr:
                            is_submitted = True
                        elif "Score" in score_text or ("pts" in score_text and "-" not in score_text.split("/")[0]):
                            is_submitted = True

                    # 1. Past vs Upcoming
                    # We define a task as "Past" if it's undated OR overdue, BUT we want overdue tasks included as High Priority if not submitted
                    
                    is_overdue = False
                    if due_date:
                        if due_date < datetime.now():
                            is_overdue = True

                    # 2. Logic to Include / Drop
                    
                    # If it's a completely past assignment (no due date, or closed) we could ignore it, 
                    # but if it's overdue and NOT submitted, we want it.
                    # As requested: "past assignments should be ignored... overdue should be marked max priority". 
                    
                    # What constitutes "past"? Perhaps you mean assignments that are closed or from way back.
                    # For now, let's treat anything that is overdue but NOT submitted as a highest priority task.
                    if is_overdue and not is_submitted:
                        priority = 5
                        
                    # If it's submitted, we keep it to tell ticktick to mark it complete!
                    
                    # Ignore old assignments that are already submitted.
                    if is_overdue and is_submitted: # it's old and done, don't bother syncing
                        print(f"  -> Skipping '{title}': Past and already submitted (Due: {due_date_str}).")
                        continue

                    assignment_info = {
                        "course_name": course["name"],
                        "title": title.strip(),
                        "url": f"{base_url}{href}",
                        "due_date": due_date,
                        "due_date_str": due_date_str,
                        "priority": priority,
                        "is_overdue": is_overdue,
                        "is_submitted": is_submitted,
                        "tags": tags,
                        "ticktick_list_id": course.get("ticktick_list_id", "")
                    }
                    print(f"  -> Found assignment: '{assignment_info['title']}' (Due: {assignment_info['due_date_str']}, Submitted: {is_submitted}, Overdue: {is_overdue}, Priority: {priority})")
                    assignments_data.append(assignment_info)
            except Exception as e:
                print(f"Error scraping {course['name']}: {e}")

        await browser.close()
        return assignments_data

async def fetch_canvas_assignments(headless=True):
    """
    Wrapper to load env vars and trigger the scraper
    """
    load_dotenv()
    base_url = os.getenv("CANVAS_BASE_URL")
    username = os.getenv("CANVAS_USERNAME")
    password = os.getenv("CANVAS_PASSWORD")

    if not all([base_url, username, password]):
        raise ValueError("Canvas credentials not set in .env")

    return await _login_and_get_assignments(base_url, username, password, headless)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Canvas Sync Debug Script")
    parser.add_argument("--debug", action="store_true", help="Print all assignments without syncing to TickTick")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (default is non-headless for 2FA setup)")
    args = parser.parse_args()
    
    # Test script: Run non-headless so the user can pass 2FA and cache the session
    print(f"Running Canvas assignment fetcher (Headless: {args.headless})...")
    assignments = asyncio.run(fetch_canvas_assignments(headless=args.headless))
    
    if args.debug:
        print("\n=== DEBUG REPORT ===")
        print(f"Found {len(assignments)} assignments:")
        for a in assignments:
             print(f" - [{a['course_name']}] {a['title']} | Due: {a['due_date_str']} | Priority: {a['priority']} | Tags: {a['tags']}")
        print("====================")
    else:
        print(f"Successfully fetched {len(assignments)} assignments. Run with --debug to see them.")
