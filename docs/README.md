# Workflow PlantUML

Workflow figure is generated using [PlantUML](https://plantuml.com/ditaa) with
the [ditaa](https://ditaa.sourceforge.net).
To generate it yourself, the easiest way is to install the
[PlantUML plugin in VS Code](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml)
(with the prerequisite installed), open the file and click preview.

If you don't want to install the dependencies locally, you can use the following
settings to make the preview work with a remote render:

```json
"plantuml.render": "PlantUMLServer",
"plantuml.server": "https://www.plantuml.com/plantuml",
```

[ASCIIFlow](https://asciiflow.com/#/) is a helpful tool to edit the source code.
