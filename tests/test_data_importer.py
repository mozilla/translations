import json
import os

import pytest
import zstandard as zstd
from fixtures import DataDir, en_sample, get_mocked_downloads, ru_sample
from pipeline.data import dataset_importer
from pipeline.data.dataset_importer import run_import

SRC = "ru"
TRG = "en"
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

hplt_translations = {
    "en": [
        "If you are having problems with your HP Computer, the article below will help determine if the problem is with your HP Drivers. Downloading the latest Driver releases helps resolve Driver conflicts and improve your computer's stability and performance. Updates are recommended for all Windows 10, 8, Windows 7, XP and Vista users.\n",
        "This webpage shows information on what a issue is and the various issues that may occurs when no permission to write document. Several issue are easy to fix, but others are not, so you need to read our article to find proper methods, so, our article is very necessary for you to read.\n",
        "0x8004D00D XACT_E_NOTCURRENT The transaction failed to commit due to the failure of optimistic concurrency control in at least one of the resource managers.\n",
        "5. Download the driver which you want. There may be numerous variations shown. Pick the latest one. If these doesn't work, the easiest way you can see.\n",
        '3. Depending on your view options either click on "uninstall a program" or "program and features". 4. When the programs and features window opens select the program your want to uninstall from the list and click on the "Uninstall" button. 5. Confirm that you want to uninstall a program by clicking on the "Yes" button.\n',
        "2.Follow the instructions in the preceding procedure to update drivers. 3.Click Search automatically for updated driver software. 4. If below message popped up, your driver is already the latest driver and there is no need to update. 5. If a new driver is found, please follow the instruction to install it and restart your computer. Description Compatibility\n",
        "Show the most common Drivers below\n",
        "HP Drivers Download Utility was created to save your time resolving driver problems by providing you with a single, automatic tool.\n",
        "All your drivers just in a minute!\n",
        "Have you encountered and don't know how to resolve issue? This guide show information on most usual lead to for problem, we hope your PC can in order after reading this.\n",
    ],
    "ru": [
        "Мы прошли проверку временем, и стали одним из ведущих операторов на рынке России в поставках металла как со склада, так и напрямую с металлургических комбинатов на объекты наших клиентов. Мы предлагаем оптимальные варианты поставок металла вашему предприятию в нужные сроки, в требуемом объеме, с необходимым качеством.\n",
        "Именно так мы видим нашу историю, – 17 лет ежедневного упорного труда «вопреки, а не благодаря», - формируя рынок и меняя компанию. Это история, которую мы пишем и сегодня, - принимая решения и воплощая их в жизнь. Это история про отношения и про внутренние рекорды. История о том, как складывать пазл внутри компании, достигать, и кропотливо создавать следующий пазл. История о том, как этические ценности становятся генетическим кодом компании. История строительства умного бизнеса. История создания своей бизнес-модели, во главу которой поставлен клиент и его потребности. Мы не останавливаемся, мы уже работаем в будущем.\n",
        "«быть конкурентоспособней» – мы становимся полностью независимым металлотрейдером, это решение во многом предопределило наше острое восприятие рынка и потребностей наших клиентов.\n",
        "«быть лучше» – мы значительно улучшили предлагаемый клиентам ассортимент, - мы начинаем работать со всеми ведущими металлургическим комбинатами России и зарубежья.\n",
        "«быть в гармонии» – сетевое развитие становится стратегической целью компании, в её рамках клиентская перспектива становится ключевым компонентом.\n",
        "Поэтому никакой критической ситуации сейчас нет, как и необходимости вводить какие-либо ограничения на федеральном уровне. Но рекомендуется следовать стандартным мерам предосторожности — носить маски в общественных местах, соблюдать гигиену рук. Это крайне важно для уязвимых категорий граждан. Также следует вакцинироваться, если подошло время.\n",
        "Отмечу, что сегодняшний рост в большей степени связан со сменой циркулирующих субвариантов вируса, а не с сезоном летних отпусков. Ведь летом люди проводят больше времени на открытом воздухе, где риски инфицирования не столь высоки. Хотя в некоторых случаях повышенная мобильность, смена климата и часовых поясов всё же могут приводить к ослаблению иммунитета и способствовать развитию инфекции. Это многие ощущали на себе в период после летних отпусков.\n",
        "— Итальянские исследователи пришли к выводу, что к каждому 20-му, кто переболел коронавирусом, возможно, никогда больше не вернутся обоняние и вкус. О подобных осложнениях также говорят российские специалисты. Вызывает ли постковид новые, более лёгкие варианты коронавируса? — Постковидный синдром продолжает регистрироваться у людей по всему миру. Считается, что он встречается у одного из восьми взрослых. В отдельных случаях при постковидном синдроме наблюдаются признаки поражения сердечно-сосудистой системы, почек, метаболические нарушения. У некоторых пациентов стойкая потеря вкуса и запаха отмечается уже на протяжении двух лет.\n",
        "Главным фактором прекращения пандемии COVID-19 станет коллективный иммунитет, его можно достичь за счёт вакцинации. Об этом в интервью...\n",
        "Сложно сказать, восстановится ли окончательно вкус или обоняние у тех, кто испытывает проблемы с ними после перенесённого COVID-19. Ещё в доковидную эпоху было показано, что 1,5% населения имеют проблемы с обонянием. После 50 лет их ощущают более 50% населения, а после 80 лет — более 80%. Поэтому если предрасположенность к этим нарушениям у человека была и до COVID-19, то такие нарушения после перенесённой инфекции могут продолжаться неопределённо долго.\n",
    ],
}

hplt_stats = {
    "en": {
        "shards": {
            "description": "How many shards were sampled from. Each shard contains a subset of the total datasets available.",
            "filtered": 1,
            "kept": 1,
            "visited": 2,
        },
        "visited_lines": {
            "description": "How many lines were visited and kept from the HPLT documents.",
            "filtered": 1517,
            "kept": 208,
            "visited": 1725,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 15,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 27,
        },
        "final_lines": {
            "description": "How many lines were actually written. Smaller lines will be combined together.",
            "value": 100,
        },
    },
    "ru": {
        "shards": {
            "description": "How many shards were sampled from. Each shard contains a subset of the total datasets available.",
            "filtered": 1,
            "kept": 1,
            "visited": 2,
        },
        "visited_lines": {
            "description": "How many lines were visited and kept from the HPLT documents.",
            "filtered": 2192,
            "kept": 164,
            "visited": 2356,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 22,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 33,
        },
        "final_lines": {
            "description": "How many lines were actually written. Smaller lines will be combined together.",
            "value": 100,
        },
    },
}


def add_fake_alignments(corpus):
    corpus_and_aln = []
    for line in corpus:
        parts = line.split("\t")
        src_sent, trg_sent = parts[0], parts[1]
        min_len = min(len(src_sent.split()), len(trg_sent.split()))
        aln = " ".join([f"{idx}-{idx}" for idx in range(min_len)])
        corpus_and_aln.append(f"{line}\t{aln}")

    return corpus_and_aln


# it's very slow to download and run BERT on 2000 lines
dataset_importer.add_alignments = add_fake_alignments


def read_lines(path):
    with zstd.open(path, "rt") as f:
        return f.readlines()


def is_title_case(text):
    return all((word[0].isupper() or not word.isalpha()) for word in text.split())


def is_title_lines(src_l, trg_l, aug_src_l, aug_trg_l):
    return is_title_case(aug_src_l) and is_title_case(aug_trg_l)


def is_upper_case(text):
    return all((word.isupper() or not word.isalpha()) for word in text.split())


def is_upper_lines(src_l, trg_l, aug_src_l, aug_trg_l):
    return is_upper_case(aug_src_l) and is_upper_case(aug_trg_l)


def only_src_is_different(src_l, trg_l, aug_src_l, aug_trg_l):
    return src_l != aug_src_l and trg_l == aug_trg_l


def src_and_trg_are_different(src_l, trg_l, aug_src_l, aug_trg_l):
    return src_l != aug_src_l and trg_l != aug_trg_l


def aug_lines_are_not_too_long(src_l, trg_l, aug_src_l, aug_trg_l):
    return (
        len(src_l) <= len(aug_src_l)
        and len(trg_l) <= len(aug_trg_l)
        # when Tags modifier is enabled with 1.0 probability it generates too many noise insertions in each sentence
        # the length ratio can still be high for one word sentences
        and len(aug_src_l) < len(src_l) * 4
        and len(aug_trg_l) < len(trg_l) * 4
    )


def all_len_equal(*items):
    return len(set(items)) == 1


def twice_longer(src, trg, aug_src, aug_trg):
    return src * 2 == aug_src and trg * 2 == aug_trg


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_data_importer")


@pytest.mark.parametrize(
    "importer,dataset",
    [
        ("mtdata", "Neulab-tedtalks_test-1-eng-rus"),
        ("opus", "ELRC-3075-wikipedia_health_v1"),
        ("flores", "dev"),
        ("sacrebleu", "wmt19"),
        ("url", "gcp_pytest-dataset_a0017e"),
    ],
)
def test_basic_corpus_import(importer, dataset, data_dir):
    data_dir.run_task(
        f"dataset-{importer}-{dataset}-en-ru",
        env={
            "WGET": os.path.join(CURRENT_FOLDER, "fixtures/wget"),
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
    )

    prefix = data_dir.join(f"artifacts/{dataset}")
    output_src = f"{prefix}.ru.zst"
    output_trg = f"{prefix}.en.zst"

    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    assert len(read_lines(output_src)) > 0
    assert len(read_lines(output_trg)) > 0


mono_params = [
    ("news-crawl", "en", "news_2021", [0, 1, 4, 6, 3, 7, 5, 2]),
    ("news-crawl", "ru", "news_2021", [0, 1, 4, 6, 3, 7, 5, 2]),
    ("url", "en", "gcp_pytest-dataset_en_cdd0d7", [2, 1, 5, 4, 0, 7, 6, 3]),
    ("url", "ru", "gcp_pytest-dataset_ru_be3263", [5, 4, 2, 0, 7, 1, 3, 6]),
]  # fmt: skip


@pytest.mark.parametrize(
    "importer,language,dataset,sort_order",
    mono_params,
    ids=[f"{d[0]}-{d[1]}" for d in mono_params],
)
def test_mono_source_import(importer, language, dataset, sort_order, data_dir):
    data_dir.run_task(
        f"dataset-{importer}-{dataset}-{language}",
        env={
            "WGET": os.path.join(CURRENT_FOLDER, "fixtures/wget"),
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
    )

    prefix = data_dir.join(f"artifacts/{dataset}")
    mono_data = f"{prefix}.{language}.zst"

    data_dir.print_tree()

    sample = {
        "en": en_sample,
        "ru": ru_sample,
    }

    sample_lines = sample[language].splitlines(keepends=True)

    assert os.path.exists(mono_data)
    source_lines = list(read_lines(mono_data))
    assert [
        source_lines.index(line) for line in sample_lines
    ] == sort_order, "The data is shuffled."


@pytest.mark.parametrize(
    "language",
    ["ru", "en"],
)
def test_mono_hplt(language, data_dir: DataDir):
    dataset = "mono_v1_2"
    data_dir.print_tree()
    max_sentences = 100

    data_dir.run_task(
        f"dataset-hplt-{dataset}-{language}",
        env={
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
        extra_args=["--max_sentences", str(max_sentences)],
    )
    data_dir.print_tree()

    lines = read_lines(data_dir.join(f"artifacts/{dataset}.{language}.zst"))
    assert lines[:10] == hplt_translations[language]

    assert len(lines) == max_sentences

    assert (
        json.loads(data_dir.load(f"artifacts/{dataset}.{language}.stats.json"))
        == hplt_stats[language]
    )


@pytest.mark.parametrize(
    "params",
    [
        ("sacrebleu_aug-upper_wmt19", is_upper_lines, all_len_equal, None, 1.0, 1.0),
        ("sacrebleu_aug-title_wmt19", is_title_lines, all_len_equal, None, 1.0, 1.0),
        # there's a small chance for the string to stay the same
        ("sacrebleu_aug-typos_wmt19", only_src_is_different, all_len_equal, None, 0.95, 1.0),
        # noise modifier generates extra lines
        ("sacrebleu_aug-noise_wmt19", lambda x: True, twice_longer, None, 0.0, 0.0),
        (
            "sacrebleu_aug-inline-noise_wmt19",
            src_and_trg_are_different,
            all_len_equal,
            aug_lines_are_not_too_long,
            # we reduce probability otherwise it generates too much noise in each sentence
            0.4,
            0.7,
        ),
    ],
    ids=["upper", "title", "typos", "noise", "inline-noise"],
)
def test_specific_augmentation(params, data_dir):
    dataset, check_is_aug, check_corpus_len, check_lines, min_rate, max_rate = params
    original_dataset = "sacrebleu_wmt19"
    prefix_aug = data_dir.join(dataset)
    prefix_original = data_dir.join(original_dataset)
    output_src = f"{prefix_aug}.{SRC}.zst"
    output_trg = f"{prefix_aug}.{TRG}.zst"
    original_src = f"{prefix_original}.{SRC}.zst"
    original_trg = f"{prefix_original}.{TRG}.zst"
    run_import("corpus", original_dataset, prefix_original, src=SRC, trg=TRG)

    run_import("corpus", dataset, prefix_aug, src=SRC, trg=TRG)

    data_dir.print_tree()
    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    src, trg, aug_src, aug_trg = (
        read_lines(original_src),
        read_lines(original_trg),
        read_lines(output_src),
        read_lines(output_trg),
    )
    assert check_corpus_len(len(src), len(trg), len(aug_src), len(aug_trg))
    if len(src) == len(aug_src):
        aug_num = 0
        for lines in zip(src, trg, aug_src, aug_trg):
            if check_lines:
                assert check_lines(*lines)
            if check_is_aug(*lines):
                aug_num += 1
        rate = aug_num / len(src)
        assert rate >= min_rate
        assert rate <= max_rate


def test_augmentation_mix(data_dir):
    dataset = "sacrebleu_aug-mix_wmt19"
    original_dataset = "sacrebleu_wmt19"
    prefix = data_dir.join(dataset)
    prefix_original = data_dir.join(original_dataset)
    output_src = f"{prefix}.{SRC}.zst"
    output_trg = f"{prefix}.{TRG}.zst"
    original_src = f"{prefix_original}.{SRC}.zst"
    original_trg = f"{prefix_original}.{TRG}.zst"
    run_import("corpus", original_dataset, prefix_original, src=SRC, trg=TRG)

    run_import("corpus", dataset, prefix, src=SRC, trg=TRG)

    AUG_MAX_RATE = 0.35
    AUG_MIN_RATE = 0.01
    data_dir.print_tree()
    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    src, trg, aug_src, aug_trg = (
        read_lines(original_src),
        read_lines(original_trg),
        read_lines(output_src),
        read_lines(output_trg),
    )
    len_noise_src = len(aug_src) - len(src)
    len_noise_trg = len(aug_trg) - len(trg)
    # check noise rate
    for noise, original in [(len_noise_src, len(src)), (len_noise_trg, len(trg))]:
        noise_rate = noise / original
        assert noise_rate > AUG_MIN_RATE
        assert noise_rate < AUG_MAX_RATE

    # check augmentation rate without noise
    for aug, original in [(aug_src, src), (aug_trg, trg)]:
        len_unchanged = len(set(aug).intersection(set(original)))
        len_original = len(original)
        aug_rate = (len_original - len_unchanged) / len(original)
        assert aug_rate > AUG_MIN_RATE
        assert aug_rate < AUG_MAX_RATE


def test_augmentation_mix_zh(data_dir):
    SRC = "zh"
    TRG = "en"

    dataset = "sacrebleu_aug-mix-cjk_wmt19"
    original_dataset = "sacrebleu_wmt19"
    prefix = data_dir.join(dataset)
    prefix_original = data_dir.join(original_dataset)
    output_src = f"{prefix}.{SRC}.zst"
    output_trg = f"{prefix}.{TRG}.zst"
    original_src = f"{prefix_original}.{SRC}.zst"
    original_trg = f"{prefix_original}.{TRG}.zst"
    run_import("corpus", original_dataset, prefix_original, src=SRC, trg=TRG)

    run_import("corpus", dataset, prefix, src=SRC, trg=TRG)

    AUG_MAX_RATE = 0.35
    AUG_MIN_RATE = 0.01
    data_dir.print_tree()
    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    src, trg, aug_src, aug_trg = (
        read_lines(original_src),
        read_lines(original_trg),
        read_lines(output_src),
        read_lines(output_trg),
    )
    len_noise_src = len(aug_src) - len(src)
    len_noise_trg = len(aug_trg) - len(trg)
    # check noise rate
    for noise, original in [(len_noise_src, len(src)), (len_noise_trg, len(trg))]:
        noise_rate = noise / original
        assert noise_rate > AUG_MIN_RATE
        assert noise_rate < AUG_MAX_RATE

    # check augmentation rate without noise
    for aug, original in [(aug_src, src), (aug_trg, trg)]:
        len_unchanged = len(set(aug).intersection(set(original)))
        len_original = len(original)
        aug_rate = (len_original - len_unchanged) / len(original)
        assert aug_rate > AUG_MIN_RATE
        assert aug_rate < AUG_MAX_RATE
