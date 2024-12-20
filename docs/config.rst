CLI Configuration
=================

.. module:: instructlab.configuration

InstructLab's configuration is read from the ``$/XDG_CONFIG_DIR/instructlab/config.yaml`` file.
The configuration is handled and valided by a `Pydantic <https://docs.pydantic.dev/>`_ schema.

.. autopydantic_model:: Config
   :model-show-json: True

General
-------

.. autopydantic_model:: _general

Metadata
--------

.. autopydantic_model:: _metadata

ilab model chat
---------------

.. autopydantic_model:: _chat

ilab model evaluate
-------------------

.. autopydantic_model:: _evaluate
.. autopydantic_model:: _mmlu
.. autopydantic_model:: _mmlubranch
.. autopydantic_model:: _mtbench
.. autopydantic_model:: _mtbenchbranch

ilab data generate
------------------

.. autopydantic_model:: _generate

ilab model serve
----------------

.. autopydantic_model:: _serve
.. autopydantic_model:: _serve_llama_cpp
.. autopydantic_model:: _serve_vllm
.. autopydantic_model:: _serve_server

ilab model train
----------------

.. autopydantic_model:: _train
