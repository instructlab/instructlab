# Taxonomy input formats

## Static `qna.yaml`

The CLI supports a number of formats for taxonomy definitions. The default mode
is a static `qna.yaml` file, populated by contributors of a new taxonomy with
the question-answer pairs.

The exact format of the file should be negotiated between the `cli` and
`taxonomy` repositories.

## Dynamic `qna.yaml` generation

Alternatively, dynamic taxonomies can be used to generate question-answer pairs
(and the rest of the `qna.yaml` file) programmatically. This mode can be useful
when generating a large number of seed samples using some basic rule that can
be better expressed with a program than by having humans manually type them in.

The general idea here is to run a containerized program that would spit out the
`qna.yaml` file when executed. Then the file is picked up by the CLI for sample
generation purposes, same way as it would do with a statically defined
`qna.yaml` file.

To define a dynamic taxonomy, put a `Containerfile` (or `Dockerfile`) in a
taxonomy directory. The container definition will be built by CLI and executed
with a temporary directory mounted into it under `/out` location. The container
command (`CMD`) is then expected to put a complete `qna.yaml` file with the
intended question-answer pairs and full metadata under `/out/qna.yaml`, from
where the CLI will pick the file up and pass it as seed samples for further
generation steps.
