from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
cameras = graph.get_input_devices()

print("Available camera devices:")

for index, camera in enumerate(cameras):
    print(f"{index}: {camera}")
    