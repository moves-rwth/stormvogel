"""Communication from Javascript/HTML to IPython/Jupyter lab using a local server and requests."""

import http.server
import threading
import urllib
import IPython.display as ipd
from time import sleep

result = ""
received_result = False


class JSMessenger:
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

        class LocalServer(http.server.BaseHTTPRequestHandler):
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
                global result, received_result
                result = urllib.parse.unquote(self.path[1:])
                received_result = True
                print(urllib.parse.unquote(self.path[1:]))

        self.web_server: http.server.HTTPServer = http.server.HTTPServer(
            ("localhost", self.server_port), LocalServer
        )
        thr = threading.Thread(target=self.__run_server)
        thr.start()

    def request(self, js: str):
        """Return the result of a single line of Javascript. Waits for at most one second, then fails."""
        global result, received_result
        received_result = False
        # Request the info
        ipd.display(
            ipd.HTML(
                f"<script>fetch('http://127.0.0.1:{self.server_port}/' + {js})</script>"
            )
        )
        # Wait until result is received.
        max_tries = 5
        while max_tries > 0 and not received_result:
            sleep(0.2)
            max_tries -= 1
        # Handle case of failure

        if not received_result:
            received_result = False
            return "FAILURE"
        else:
            received_result = False
            return result

    def __run_server(self):
        print("Server started http://%s:%s" % ("localhost", self.server_port))
        try:
            self.web_server.serve_forever()
        except KeyboardInterrupt:
            pass
        self.web_server.server_close()
        print("Server stopped.")
