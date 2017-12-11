from wsgiref.simple_server import make_server

def hello_world(environ, start_response):
    status = '200 ok'
    headers = [('Content-type', 'text/plain')]
    start_response(status, headers)

    return

httpd = make_server('', 8005, hackton_TED)

print ("Serving on porting 8005....")

httpd.serve_forever()