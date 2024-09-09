"""Communication from Javascript/HTML to IPython/Jupyter lab using a local server and requests.
Initialization by user is not recommended. It should happen automatically when creating a visjs.Network.

Remember that THE PORT STORED IN GLOBAL VARIABLE server_port NEEDS TO BE OPEN (AND IN SOME CASES FORWARDED) FOR IT TO WORK."""

import http.server
import random
import string
import threading
import urllib
import IPython.display as ipd
import logging
from time import sleep

server_warning: bool = True
request_warning: bool = True


server_port: int = 8080
"""Change this to use a different port, BEFORE LOADING A NETWORK!!!"""


def set_port(port: int):
    """Call before creating any Network (or Visualization, or calling show.show()) to change the port.
    If you have already done one of these things, restart the kernel or call initialize_server with reset=True."""
    global server_port
    server_port = port


awaiting: dict = {}
mutex: threading.Lock = threading.Lock()
server_running: bool = False

AWAITING = "AWAITING"


class CommunicationServer:
    def __init__(self, server_port: int = 8080) -> None:
        """Run a web server in the background to receive Javascript communications.
        Warning! We don't currently account for race conditions etc.
        JSMessenger might behave unexpectedly if multiple requests are going on at the same time.

        Args:
            on_get (Callable[[str]]): Function to call when a message is received.
            host_name (str, optional): Defaults to "localhost".
            server_port (int, optional): Defaults to 8080.
        """
        self.server_port: int = server_port
        global server_running

        class InnerServer(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    bytes(
                        "<html><head><title>Javascript and IPython communications channel.</title></head>",
                        "utf-8",
                    )
                )
                self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
                self.wfile.write(bytes("<body>", "utf-8"))
                self.wfile.write(
                    bytes(
                        "<p>Javascript and IPython communications channel.</p>", "utf-8"
                    )
                )
                self.wfile.write(bytes("</body></html>", "utf-8"))
                logging.info(f"Received request: {urllib.parse.unquote(self.path)}")
                try:
                    identifier, message = urllib.parse.unquote(self.path)[1:].split(
                        "/MESSAGE/"
                    )
                    global awaiting
                    with mutex:
                        if identifier in awaiting:
                            awaiting[identifier] = message
                except ValueError:
                    logging.warning("Could not split message!")

            def log_message(self, format, *args):
                """To prevent an unwanted default log message from coming up. Not used by stormvogel."""
                pass

        self.web_server: http.server.HTTPServer = http.server.HTTPServer(
            ("localhost", self.server_port), InnerServer
        )
        thr = threading.Thread(target=self.run_server)
        thr.start()

    def request(self, js: str):
        """Return the result of a single line of Javascript. Waits for at most 2 seconds, then throws TimeoutError.
        Also waits for server to boot up if it is not finished yet.
        Should be thread safe. (I hope).
        WHEN SENDING JAVASCRIPT, DO NOT FORGET EXTRA QUOTES AROUND STRINGS."""
        global awaiting, server_running
        while not server_running:  # Wait for server to start.
            sleep(0.2)
            logging.debug("Request waiting for server to run.")
        # Random identifier of lenght 10.
        identifier = "".join(random.choices(string.ascii_letters, k=10))
        html = f"<script>fetch('http://127.0.0.1:{self.server_port}/{identifier}/MESSAGE/' + {js})</script>"
        logging.debug(f"full html: {html}")
        # Request the info
        ipd.display(ipd.HTML(html))

        awaiting[identifier] = AWAITING
        logging.info(f"Request sent for: {identifier}")
        # Wait until result is received.
        max_tries = 10
        while max_tries > 0 and awaiting[identifier] == AWAITING:
            sleep(0.2)
            max_tries -= 1
            logging.debug(f"Waiting for request result: {identifier}")
        # Handle case of failure

        result = awaiting[identifier]
        # Remove this request from the list, we no longer wait for it.
        try:
            with mutex:
                awaiting.pop(identifier)
        except KeyError:
            pass

        if result == AWAITING:
            raise TimeoutError(
                f"CommunicationServer.request did not receive result in time for request {identifier}:\n{js}"
            )
        else:
            logging.info(f"Succesfully received result of request: {identifier}")
            return result

    def run_server(self):
        """Run the server (used to put it on a thread).
        Waits 0.5 seconds and then sets global variable server_running to true.
        This is to prevent making requests too early."""
        global server_running, server_warning
        try:
            logging.info(
                f"CommunicationsServer started http://localhost:{self.server_port}"
            )
            server_running = True
            self.web_server.serve_forever()
        except KeyboardInterrupt:
            pass


server: CommunicationServer | None = None
"""Global variable holding the server used for this notebook. None if not initialized."""


def initialize_server() -> CommunicationServer | None:
    """If server is None, then create a new server and store it in global variable server.
    Use the port stored in global variable server_port.
    """
    global server, server_port, server_running
    try:
        if server is None:
            server = CommunicationServer(server_port=server_port)
        return server
    except OSError:
        logging.error("Server port taken.")
        if server_warning:
            print(f"""Stormvogel could not run an internal server to communicate between local processes on localhost:{server_port}.
Stormvogel is still usable without this, but a few visualization features might not be available. set stormvogel.communication_server.server_warning to False to ignore this message.
This might be solved as such:
1) If you already had a stormvogel notebook with a Network or Visualization or show.show in it in this lab session, change the kernel of the current notebook to be the SAME KERNEL (Top right, use kernel for preferred session and look for the name of the PREVIOUS notebook).
You can also simply restart all kernels but it might break again.
2) Port {server_port} might already be used by another process. Try changing stormvogel.communication_server.server_port and running again.
Please contact the stormvogel developpers if you keep running into issues.""")
