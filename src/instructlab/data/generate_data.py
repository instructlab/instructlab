# Standard
import logging

# First Party
from instructlab.client_utils import HttpClientParams

logger = logging.getLogger(__name__)


# handler function to be used as an intermediary so we can manipulate args
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
    process_mode,
    log_level,
):
    """Generates synthetic data to enhance your example data"""
    # First Party
    from instructlab.defaults import ILAB_PROCESS_TYPES
    from instructlab.process.process import add_process

    add_process(
        process_mode=process_mode,
        process_type=ILAB_PROCESS_TYPES.DATA_GENERATION,
        target=create_server_and_generate,
        extra_imports=[
            ("instructlab.configuration", "_serve", "_serve_vllm", "_serve_llama_cpp")
        ],
        log_level=log_level,
        serve_cfg=serve_cfg,
        model_name=model_path,
        num_cpus=num_cpus,
        num_instructions_to_generate=sdg_scale_factor,
        taxonomy=taxonomy_path,
        taxonomy_base=taxonomy_base,
        output_dir=output_dir,
        console_output=not quiet,
        endpoint_url=endpoint_url,
        api_key=api_key,
        yaml_rules=yaml_rules,
        chunk_word_count=chunk_word_count,
        server_ctx_size=server_ctx_size,
        tls_client_cert=http_client_params.get("tls_client_cert")
        if http_client_params.get("tls_client_cert") != ""
        else None,
        tls_client_key=http_client_params.get("tls_client_key")
        if http_client_params.get("tls_client_key") != ""
        else None,
        tls_client_passwd=http_client_params.get("tls_client_passwd")
        if http_client_params.get("tls_client_passwd") != ""
        else None,
        tls_insecure=http_client_params.get("tls_insecure")
        if http_client_params.get("tls_insecure") != ""
        else None,
        model_family=model_family,
        pipeline=pipeline,
        enable_serving_output=enable_serving_output,
        batch_size=batch_size,
        gpus=gpus,
        checkpoint_dir=checkpoint_dir,
        max_num_tokens=max_num_tokens,
        system_prompt=system_prompt,
        use_legacy_pretraining_format=use_legacy_pretraining_format,
        # http_client_params=http_client_params,
    )


# actual generate_data function to be kicked off in its own process/eventually be REST API endpoint
# pylint: disable=redefined-outer-name
def create_server_and_generate(
    log_level,
    serve_cfg,
    model_name,
    num_cpus,
    num_instructions_to_generate,
    taxonomy,
    taxonomy_base,
    output_dir,
    console_output,
    endpoint_url,
    api_key,
    yaml_rules,
    chunk_word_count,
    server_ctx_size,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    tls_insecure,
    model_family,
    pipeline,
    enable_serving_output,
    batch_size,
    gpus,
    checkpoint_dir,
    max_num_tokens,
    system_prompt,
    use_legacy_pretraining_format,
    log_file,
):
    backend_instance = None
    # we need to use the instructlab logger so that the libraries inherit the config we set up.
    logger = logging.getLogger("instructlab")

    # First Party
    from instructlab.log import add_file_handler_to_logger

    # when in a new process, logger configuration does not seem to be promised. Add the file handler pointing to our new log file
    # and also, reenforce the level set on the instructlab logger.
    add_file_handler_to_logger(log_file=log_file, logger=logger)
    logger.setLevel(log_level)

    # Third Party
    import openai

    # First Party
    from instructlab.client_utils import http_client

    http_client_params = HttpClientParams(
        {
            "tls_client_cert": tls_client_cert,
            "tls_client_key": tls_client_key,
            "tls_client_passwd": tls_client_passwd,
            "tls_insecure": tls_insecure,
        }
    )

    if endpoint_url:
        api_base = endpoint_url
    else:
        # First Party
        from instructlab.model.backends import backends
        from instructlab.model.backends.llama_cpp import Server as llama_cpp_server

        backend_instance = backends.select_backend(cfg=serve_cfg, model_path=model_name)
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
    # Third Party
    from instructlab.sdg.generate_data import generate_data

    # pylint: disable=ungrouped-imports
    from instructlab.sdg.utils import GenerateException

    try:
        logger.info(
            f"Generating synthetic data using '{pipeline}' pipeline, '{model_name}' model, '{taxonomy}' taxonomy, against {api_base} server"
        )
        generate_data(
            client=client,
            model_family=model_family,
            model_name=model_name,
            num_cpus=num_cpus,
            num_instructions_to_generate=num_instructions_to_generate,
            taxonomy=taxonomy,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
            console_output=console_output,
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
