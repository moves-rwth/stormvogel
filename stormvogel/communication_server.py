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
import ipywidgets as widgets
import socket

enable_server: bool = True
"""Disable if you don't want to use an internal communication server. Some features might break."""

localhost_address: str = "127.0.0.1"

min_port = 8889
max_port = 8905
port_range = range(min_port, max_port)
"""The range of ports that stromvogel uses. They should all be forwarded if you're on an http tunnel."""

server_port: int = 8888
"""Global variable storing the port that is being used by this process. Changes when initialize_server is called."""

awaiting: dict = {}
mutex: threading.Lock = threading.Lock()
server_running: bool = False
spam: widgets.Output = widgets.Output()

AWAITING = "AWAITING"


class CommunicationServer:
    def __init__(self, server_port: int = 8080) -> None:
        """Run a web server in the background to receive Javascript communications.
        Warning! We don't currently account for race conditions etc.
        JSMessenger might behave unexpectedly if multiple requests are going on at the same time.

        Args:
            server_port (int, optional): Defaults to 8080.
        """
        self.server_port: int = server_port

        class InnerServer(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    bytes(
                        "<html><head><title>Stormvogel's Javascript and IPython communication channel.</title></head>",
                        "utf-8",
                    )
                )
                self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
                self.wfile.write(bytes("<body>", "utf-8"))
                self.wfile.write(
                    bytes(
                        "<p>Stormvogel's Javascript and IPython communication channel.</p>",
                        "utf-8",
                    )
                )
                self.wfile.write(bytes("</body></html>", "utf-8"))
                logging.info(f"Received request: {urllib.parse.unquote(self.path)}")
                try:
                    identifier, message = urllib.parse.unquote(self.path)[1:].split(
                        "/MESSAGE/"
                    )
                    global awaiting, spam
                    if identifier in awaiting:
                        awaiting[identifier] = message
                    logging.info(f"Stored message {identifier}")
                    try:
                        logging.debug(awaiting[identifier])
                    except KeyError:
                        pass
                    # There was a weird bug where sometimes a request simply wouldn't work if triggered from a button.
                    # These two lines seem to fix it don't ask me why, and also don't remove them. I think it might somehow force threads to syncronize
                    # If you really want to remove them, be sure to test the code a lot of times afterwards because it might be inconsistent.
                    with spam:
                        ipd.display(awaiting)
                except ValueError:
                    logging.warning("Could not split message!")

            def log_message(self, format, *args):
                """To prevent an unwanted default log message from coming up. Not used by stormvogel."""
                pass

        self.web_server: http.server.HTTPServer = http.server.HTTPServer(
            (localhost_address, self.server_port), InnerServer
        )
        thr = threading.Thread(target=self.run_server)
        thr.start()

    def request(self, js: str):
        """Return the result of a single line of Javascript. Waits for at most 2 seconds, then throws TimeoutError.
        Also waits for server to boot up if it is not finished yet.
        Should be thread safe. (I hope).
        WHEN SENDING JAVASCRIPT, DO NOT FORGET EXTRA QUOTES AROUND STRINGS."""
        global server
        if server is None:
            raise TimeoutError("There is no server running.")

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
        max_tries = 50
        while max_tries > 0 and awaiting[identifier] == AWAITING:
            sleep(0.2)
            max_tries -= 1
            logging.debug(f"Waiting for request result: {identifier}")
            logging.debug(f"Current awaiting[identifier]: {awaiting[identifier]}")
            ipd.display(ipd.HTML(html))
        # Handle case of failure

        result = awaiting[identifier]

        if result == AWAITING:
            raise TimeoutError(
                f"CommunicationServer.request did not receive result in time for request {identifier}:\n{js}"
            )
        else:
            logging.info(f"Succesfully received result of request: {identifier}")
            try:
                awaiting.pop(identifier)
            except KeyError:
                pass
            return result

    def run_server(self):
        """Run the server (used to put it on a thread).
        Waits 0.5 seconds and then sets global variable server_running to true.
        This is to prevent making requests too early."""
        global server_running, server_warning
        try:
            logging.info(
                f"CommunicationsServer started http://{localhost_address}:{self.server_port}"
            )
            server_running = True
            self.web_server.serve_forever()
        except KeyboardInterrupt:
            pass


server: CommunicationServer | None = None
"""Global variable holding the server used for this notebook. None if not initialized."""


def __request_warning_message():
    return f"""Stormvogel succesfully started the internal communication server, but could not receive the result of a test request.
Stormvogel is still usable without this, but you will not be able to save node positions in a layout json file.
1) Restart the kernel and re-run.
2) Is the port {localhost_address}:{server_port} (from the machine where jupyterlab runs) available?
If you are working remotely, it might help to forward this port. For example: 'ssh -N -L {server_port}:{localhost_address}:{server_port} YOUR_SSH_CONFIG_NAME'.
3) You might also want to consider changing stormvogel.communication_server.localhost_address to the IPv6 loopback address if you are using IPv6.
If you cannot get the server to work, set stormvogel.communication_server.enable_server to false and re-run. This will speed up stormvogel and ignore this message.
Please contact the stormvogel developpers if you keep running into issues."""


def __server_warning_message():
    return f"""Stormvogel could not run an internal server to communicate between local processes on {localhost_address}:{server_port}.
Stormvogel is still usable without this, but you will not be able to save node positions in a layout json file.
This might be solved as such:
1) Restart the kernel and re-run.
2) Port {server_port} might already be used by another process, or even another jupyter lab kernel. Try changing stormvogel.communication_server.server_port and running again.
3) You might also want to consider changing stormvogel.communication_server.localhost_address to the IPv6 loopback address if you are using IPv6.
If you cannot get the server to work, set stormvogel.communication_server.enable_server to false and re-run. This will speed up stormvogel and ignore this message.
Please contact the stormvogel developpers if you keep running into issues."""


def __no_free_port_warning_message():
    return f"""Stormvogel could not find a free port in the range [{min_port, max_port}) to host a local process.
Stormvogel can still function without this, but you will not be able to save node positions in a layout json file.
If you have a lot of notebooks open, it might help to restart jupyter lab or close some kernels from other notebooks.
If the default range of ports does not work for you, feel free to edit stormvogel.communication_server.min_port and stormvogel.communication_server.max_port.
Please contact the stormvogel developpers if you keep running into issues."""


def is_port_free(port: int) -> bool:
    """Return true iff the specified port is free on localhost_address.
    Thanks to StackOverflow user Rugnar
    https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((localhost_address, port)) != 0


def find_free_port() -> int:
    for port_no in port_range:
        if is_port_free(port_no):
            return port_no
    return -1


def initialize_server() -> CommunicationServer | None:
    """If server is None, then create a new server and store it in global variable server.
    Use the port stored in global variable server_port.
    """
    global server, server_port, enable_server
    if not enable_server:
        return None

    output = widgets.Output()
    with output:
        print(
            "Initializing communication server and sending a test message. This might take a couple of seconds."
        )
    ipd.display(output)
    try:
        if server is None:
            server_port = find_free_port()
            if server_port == -1:
                logging.warning("Could not find free port.")
                print(__no_free_port_warning_message())
                with output:
                    ipd.clear_output()
                return None
            server = CommunicationServer(server_port=server_port)
            logging.info("Succesfully initialized server.")
            try:
                server.request("'test message'")
                logging.info("Succesfully received test message.")
            except TimeoutError:
                print(__request_warning_message())
        with output:
            ipd.clear_output()
        return server
    except OSError:
        logging.warning("Server port likely taken.")
        print(__server_warning_message())
        with output:
            ipd.clear_output()
