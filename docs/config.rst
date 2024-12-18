ilab configuration
==================

.. module:: instructlab.configuration

InstructLab's configuration is read from ``config.yaml`` file. The
configuration is handled avalided by a `Pydantic <https://docs.pydantic.dev/>`_
schema.

.. autopydantic_model:: Config
   :model-show-json: True

General
-------

.. autopydantic_model:: _general
.. autopydantic_model:: _metadata

model chat
----------

.. autopydantic_model:: _chat

model evaluate
--------------

.. autopydantic_model:: _evaluate
.. autopydantic_model:: _mmlu
.. autopydantic_model:: _mmlubranch
.. autopydantic_model:: _mtbench
.. autopydantic_model:: _mtbenchbranch

data generate
--------------

.. autopydantic_model:: _generate

model serve
-----------

.. autopydantic_model:: _serve
.. autopydantic_model:: _serve_llama_cpp
.. autopydantic_model:: _serve_vllm
.. autopydantic_model:: _serve_server

model train
-----------

.. autopydantic_model:: _train
