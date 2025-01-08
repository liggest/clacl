for SEED in 233 2025 10086
    do
        python -m clacl.run.CLSCL --config data/CLSCL/config_5.toml --seed $SEED
        # python -m clacl.run.finetune_CL --config data/finetune_CL/config_test.toml --seed $SEED
        # python -m clacl.run.finetune --config data/finetune/config_KS.toml --seed $SEED
        # python -m clacl.run.finetune --config data/finetune/config_IC.toml --seed $SEED
        # python -m clacl.run.finetune --config data/finetune/config_ER.toml --seed $SEED
        # python -m clacl.run.finetune --config data/finetune/config_AcC.toml --seed $SEED
        # python -m clacl.run.finetune --config data/finetune/config_LID.toml --seed $SEED
    done
