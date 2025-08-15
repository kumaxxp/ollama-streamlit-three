from app.core.dialogue_controller import DialogueController, DialogueConfig


def main():
    cfg = DialogueConfig(
        theme="テスト",
        agent1_name="high_school_girl_optimistic",
        agent2_name="office_worker_tired",
        max_turns=2,
        director_config={"model": "gemma3:4b", "check_interval": 1},
        model_params={
            "model": "qwen2.5:7b",
            "agent1_temperature": 0.6,
            "agent2_temperature": 0.8,
        },
    )

    c = DialogueController()
    c.initialize_session(cfg)

    # ダミー履歴を追加
    c._update_history(cfg.agent1_name, "はじめまして。今日は何について話す？")
    c._update_history(cfg.agent2_name, "仕事と生活のバランスについてどう思います？")

    # Director 分析実行
    res = c._perform_analysis()
    print("ANALYSIS:", res)


if __name__ == "__main__":
    main()
