# Third Party
from yaml.resolver import Resolver

# PyYAML implements YAML 1.1, which makes all kinds of implicit conversions
# that may affect how seed data is loaded into cli.
#
# To avoid this, monkey patch PyYAML Resolver to not convert Yes/No to
# True/False.
#
# See: https://github.com/yaml/pyyaml/issues/116
for resolver in Resolver.yaml_implicit_resolvers:
    # only remove resolvers for on/off/yes/no
    if resolver.lower() not in "nyo":
        continue
    Resolver.yaml_implicit_resolvers[resolver] = [
        x
        for x in Resolver.yaml_implicit_resolvers[resolver]
        if x[0] != "tag:yaml.org,2002:bool"
    ]
