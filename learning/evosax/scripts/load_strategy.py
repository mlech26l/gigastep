es_ego_logging = ESLog(
                            num_dims=train_param_ego_reshaper.total_params,
                            num_generations=num_generations,
                            top_k=5,
                            maximize=True,
                        )
                        es_ego_log = es_ego_logging.initialize()