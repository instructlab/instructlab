# Standard
import logging

# Third Party
import openai

# First Party
from instructlab.client_utils import HttpClientParams, http_client

logger = logging.getLogger(__name__)


def gen_data(
    serve_cfg,
    model_path,
    num_cpus,
    sdg_scale_factor,
    taxonomy_path,
    taxonomy_base,
    output_dir,
    quiet,
    endpoint_url,
    api_key,
    yaml_rules,
    chunk_word_count,
    server_ctx_size,
    http_client_params: HttpClientParams,
    model_family,
    pipeline,
    enable_serving_output,
    batch_size,
    gpus,
    checkpoint_dir,
    max_num_tokens,
    system_prompt,
    use_legacy_pretraining_format,
):
    """Generates synthetic data to enhance your example data"""
    # Third Party
    from instructlab.sdg.generate_data import generate_data

    # pylint: disable=ungrouped-imports
    from instructlab.sdg.utils import GenerateException

    backend_instance = None

    if endpoint_url:
        api_base = endpoint_url
    else:
        # First Party
        from instructlab.model.backends import backends
        from instructlab.model.backends.llama_cpp import Server as llama_cpp_server

        backend_instance = backends.select_backend(cfg=serve_cfg, model_path=model_path)
        if (
            backend_instance.get_backend_type() is not backends.VLLM
            and gpus is not None
        ):
            logger.debug(
                "Cannot specify '--gpus' with a llama-cpp backend, ignoring this flag."
            )

        try:
            # Run the backend server
            api_base = backend_instance.run_detached(
                http_client=http_client(http_client_params),
                background=not enable_serving_output,
                foreground_allowed=True,
                max_startup_retries=1,
            )
        except Exception as exc:
            raise ValueError(f"Failed to start server: {exc}") from exc

        # disable batching when running with the local llama.cpp server
        if isinstance(backend_instance, llama_cpp_server):
            if batch_size is not None:
                logger.warning(
                    "Disabling SDG batching - unsupported with llama.cpp serving"
                )
            batch_size = 0

    client = openai.OpenAI(
        base_url=api_base, api_key=api_key, http_client=http_client(http_client_params)
    )

    try:
        logger.info(
            f"Generating synthetic data using '{pipeline}' pipeline, '{model_path}' model, '{taxonomy_path}' taxonomy, against {api_base} server"
        )
        generate_data(
            client=client,
            model_family=model_family,
            model_name=model_path,
            num_cpus=num_cpus,
            num_instructions_to_generate=sdg_scale_factor,
            taxonomy=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
            console_output=not quiet,
            yaml_rules=yaml_rules,
            chunk_word_count=chunk_word_count,
            server_ctx_size=server_ctx_size,
            pipeline=pipeline,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            max_num_tokens=max_num_tokens,
            system_prompt=system_prompt,
            use_legacy_pretraining_format=use_legacy_pretraining_format,
        )
    except GenerateException as exc:
        raise ValueError(
            f"Generating dataset failed with the following error: {exc}"
        ) from exc
    finally:
        if backend_instance is not None:
            backend_instance.shutdown()
