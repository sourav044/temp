import time 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException

# Initialize Chrome
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Constants
URL = "https://patnahighcourt.gov.in/?p=3"
START = 1950
END = START + 1000
YEAR = "2025"
VALUE = "6"
TARGET_TEXT = "45. Quashing D ORDER OF COGNIZANCE UNDER SECTION 498-A OR 304-B IPC."
FOUND = []

# Open URL once before loop
driver.get(URL)
time.sleep(5)

# Clear output file at start
open("matched_cases.txt", "w").close()

for case_no in range(START, END):
    # Step 1
    Select(driver.find_element(By.ID, "ctl00_MainContent_ddlCTypeCri")).select_by_value(VALUE)

    # Step 2
    case_no_input = driver.find_element(By.ID, "ctl00_MainContent_txtCaseNoCri")
    case_no_input.clear()
    case_no_input.send_keys(str(case_no))

    # Step 3
    case_year_input = driver.find_element(By.ID, "ctl00_MainContent_txtCaseYearCri")
    case_year_input.clear()
    case_year_input.send_keys(YEAR)

    print(f"🕒 Waiting 10s to solve CAPTCHA for case {case_no}...")
    time.sleep(10)

    # Step 4
    driver.find_element(By.ID, "ctl00_MainContent_btnSearchCri").click()
    time.sleep(5)

    # Step 5
    try:
        result_text = driver.find_element(By.ID, "ctl00_MainContent_fvCaseStatusCri_Label23").text.strip()
        if TARGET_TEXT in result_text:
            print(f"✅ Found match for case: {case_no}")
            FOUND.append(case_no)
            # Save immediately to file
            with open("matched_cases.txt", "a") as file:
                file.write(f"{case_no}\n")
        else:
            print(f"❌ No match for case: {case_no}")
    except NoSuchElementException:
        print(f"⚠️ Span not found for case: {case_no}")

    # Go back to main form
    driver.back()
    time.sleep(3)

# Final output
print("\n✅ DONE. Total Matched Cases:", len(FOUND))


print("📁 All matched case numbers saved to 'matched_cases.txt'")

driver.quit()
