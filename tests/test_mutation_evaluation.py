from eval.mutation_cases import MUTATION_CASES


def test_all_registered_mutations_are_detected() -> None:
    assert MUTATION_CASES

    failures = []

    for mutation in MUTATION_CASES:
        detected = mutation.check()

        if not detected:
            failures.append(
                f"{mutation.mutation_id}: "
                f"{mutation.description}"
            )

    assert not failures, (
        "Mutation safeguards failed to detect: "
        + "; ".join(failures)
    )


def test_mutation_ids_are_unique() -> None:
    ids = [
        mutation.mutation_id
        for mutation in MUTATION_CASES
    ]

    assert len(ids) == len(set(ids))


def test_every_mutation_has_description() -> None:
    for mutation in MUTATION_CASES:
        assert mutation.description.strip()


def test_every_mutation_has_callable_check() -> None:
    for mutation in MUTATION_CASES:
        assert callable(mutation.check)