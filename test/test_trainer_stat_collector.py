from fine_tuning.trainer_stat_collector import TrainerStatCollector


def test_text_fields():
    collector = TrainerStatCollector(train_paramers=None)
    ref_text_field = {"test1": "body of test 1", "test2": "body of test 2"}

    for key in ref_text_field:
        collector.add_text_field(key, ref_text_field[key])

    for key in ref_text_field:
        assert collector.get_text_field(key) == ref_text_field[key]


def test_collect_train_metrics():
    collector = TrainerStatCollector(train_paramers=None)
    test_arguments = [
        {"logs": {"loss": 3.066, "learning_rate": 6.666666666666667e-06, "epoch": 2.6}},
        {"logs": {"loss": 3.0888, "learning_rate": 3.333333333333333e-06, "epoch": 2.8}},
        {"logs": {"loss": 3.0454, "learning_rate": 0.0, "epoch": 3.0}},
    ]

    for log in test_arguments:
        collector.on_log(args=None, control=None, state=None, **log)

    metrics = collector.train_metrics

    losses = metrics["loss"]
    print(f"{losses =}")
    assert list(metrics["loss"]) == [3.066, 3.0888, 3.0454]
    assert list(metrics["epoch"]) == [2.6, 2.8, 3.0]
