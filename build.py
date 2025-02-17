import json

with open("replay.json", "r") as file:
    json_data = json.load(file)

json_data_str = json.dumps(json_data).replace("</", "<\\/")

html_template = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="https://s3vis.lux-ai.org/eye.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />

    <title>Lux Eye S3</title>
    <script>
    window.episode = {json_data_str}
    </script>
    <script type="module" crossorigin src="https://s3vis.lux-ai.org/index.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
"""

with open("index.html", "w") as file:
    file.write(html_template)
