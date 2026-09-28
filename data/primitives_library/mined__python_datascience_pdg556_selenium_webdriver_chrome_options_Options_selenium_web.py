# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg556::selenium.webdriver.chrome.options.Options+selenium.webdriver.chrome.service.Service
# name: selenium_primitive
# summary: Uses selenium.webdriver.chrome.options.Options, selenium.webdriver.chrome.service.Service across 2 repos
# anchor_symbols: ['selenium.webdriver.chrome.options.Options', 'selenium.webdriver.chrome.service.Service']
# observed in 2 repos: ['pgalko__BambooAI', 'starpig1129__DATAGEN']...

# --- from pgalko__BambooAI::bambooai/google_search.py::SearchEngine.__init__ ---
def __init__(self):
        webdriver_path = os.environ.get('SELENIUM_WEBDRIVER_PATH')
        if webdriver_path and webdriver_path.strip():
            self.webdriver_path = os.path.normpath(webdriver_path)
        else:
            self.webdriver_path = None
        self.driver = None
        self.headless = True
        
        if self.webdriver_path:
            # Initialize Selenium WebDriver if path is provided
            self.service = ChromeService(executable_path=self.webdriver_path)
            self.options = Options()
            if self.headless:
                self.options.add_argument("--headless")
            self.options.add_argument("--disable-gpu")
            self.options.add_argument("--no-sandbox")
            self.options.add_argument("--disable-dev-shm-usage")

            # Set up logging preferences to suppress console messages
            self.options.add_argument("--log-level=3")
            self.options.add_experimental_option('excludeSwitches', ['enable-logging'])
            self.options.add_experimental_option('excludeSwitches', ['enable-automation'])
            self.options.add_experimental_option('useAutomationExtension', False)

            # Optionally, you can set the logging level for the browser specifically
            self.options.set_capability('goog:loggingPrefs', {
                'browser': 'OFF', 
                'driver': 'OFF', 
                'performance': 'OFF', 
                'server': 'OFF'
                })

# --- from starpig1129__DATAGEN::src/tools/internet.py::google_search ---
def google_search(query: Annotated[str, "The search query to use"]) -> Annotated[str, "The top 5 Google search results."]:
    """
    Perform a Google search based on the given query and return the top 5 results.

    This function uses Selenium to perform a headless Google search and BeautifulSoup to parse the results.

    """
    try:
        logger.info(f"Performing Google search for query: {query}")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(CHROMEDRIVER_PATH)

        with webdriver.Chrome(options=chrome_options, service=service) as driver:
            # Set a timeout to prevent hanging on slow network
            driver.set_page_load_timeout(30)
            url = f"https://www.google.com/search?q={query}"
            logger.debug(f"Accessing URL: {url}")
            driver.get(url)
            html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')
        search_results = soup.select('.g') 
        search = ""
        for result in search_results[:5]:
            title_element = result.select_one('h3')
            title = title_element.text if title_element else 'No Title'
            snippet_element = result.select_one('.VwiC3b')
            snippet = snippet_element.text if snippet_element else 'No Snippet'
            link_element = result.select_one('a')
            link = link_element['href'] if link_element else 'No Link'
            search += f"{title}\n{snippet}\n{link}\n\n"

        logger.info("Google search completed successfully")
        return search
    except Exception as e:
        logger.error(f"Error during Google search: {str(e)}")
        return f'Error: {e}'
