import json
import os

import pytest
import zstandard as zstd
from fixtures import DataDir, en_sample, get_mocked_downloads, ru_sample, zh_sample, FIXTURES_PATH
from pipeline.data import dataset_importer
from pipeline.data.dataset_importer import run_import

SRC = "ru"
TRG = "en"
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# the first 10 lines are copied from data/tests_data/test_data_importer/artifacts/mono_v1_2.{en,ru}.zst
hplt_expected = {
    "en": """So I did get some pictures off to you this week! Yay!! I actually set a new goal for myself to take a picture everyday so we’ll see how that goes. And I took a picture with the tie for Tim’s birthday with our English class, but you can’t really see the tie so I had to take a lame one of my doing some Bible bashing. . .with the Book of Mormon. But yeah sorry those took so long to send. I had some troubles the last couple of weeks and I could not send them, but it should be good from here on out.
Oh! Before I go, tell the Brigg’s a HUGE thank you! They sent me a Christmas package which was way too much for them to do. Tell them that I am super grateful to them. That made my day. And I’m eagerly awaiting yours. =) And don’t worry about the forgotten stuff, it’s all good! I will try to get your package sent next week. I finally finished getting the stuff today. Well I love you mom! You have an awesome week! We’ll talk soon!
The best compliment we can give Debbie is that we wish we could have used her services in our home state. She has set a high standard for realtors and we were fortunate enough to have been introduced to her. We truly had a wonderful business relationship and feel like we made a personal friend for years to come.
There are many Real Estate Agents to choose from, but most of them will tell you just what they think you want to hear. For example, they will list your property for a price that they know it won’t bring. Debbie will lay the true facts on the table so you can make an informed decision. She is in the real world every day and has an active sense of values and of the market.
I had the pleasure of working with Debbie Sinagoga starting nearly 2 years ago when we had to sell our home. Debbie came in during what was a very difficult and emotional time in my life and made things easier with professionalism, integrity, attention to detail, care, compassion and humor. Selling your dream home can often be difficult. It isn’t just about the transaction but the emotions that go with it. Debbie not only understood that but was able to make the process simple and was supportive throughout.
So whether your transaction centers in Scottsdale, Carefree, Cave Creek, North Phoenix, Paradise Valley, Arcadia, or Fountain Hills we’d like to take some time to talk to you about how we can help you achieve your optimal outcome. Phone(480) 703-2299 Phone(602) 527-1985 Address 7669 E Pinnacle Peak Rd. Suite 110 Scottsdale, AZ 85255
Architecture, Beautiful Home Plan Design from Mexico: Fantastic Sitting Space In The Casa Del Viento With Grey Sofas And Glass Table On The Grey Rug Uploaded by Jutta Vevica Dietrich at Friday, May 16, 2014, the amusing Fantastic Sitting Space In The Casa Del Viento With Grey Sofas And Glass Table On The Grey Rug image above is one of several amusing picss that connected to the main article Beautiful Home Plan Design from Mexico. Taged in Grey Carpet, Orange Chair, White Wall, etc.
Make sure you look all pics in the gallery part on this Beautiful Home Plan Design from Mexico so that you obtain inspiration to construct your idea. Jutta Vevica Dietrich tries to define all of the room on the Fantastic Sitting Space In The Casa Del Viento With Grey Sofas And Glass Table On The Grey Rug of this article. There are several pics that are not explained. If you are interested in finding out more information about the pics, you can contact us via the contact page below. We will help you to obtain your inspiration.
The nominations for the 70th Emmy Awards were announced this morning, and Westworld garnered an incredible 21 nominations – including Outstanding Drama Series! This falls just short of leader Game of Thrones (with 22) and helps put HBO in...
Westworld Wins at the 2017 Creative Arts Emmys! The live Emmy Awards ceremony isn’t until next Sunday, September 17th, but the Creative Arts portion of TV’s top awards took place this weekend, over two evenings at the Microsoft Theater in Los Angeles, California. Westworld was well... Westworld Roundup: Jonathan Nolan on Peter Abernathy’s Storyline; Evan Rachel Wood and Thandie Newton’s Emmy Submissions Revealed""",
    "ru": """Производитель: EVVA (Австрия)
ЗАПАТЕНТОВАННЫЙ МЕХАНИЗМ Цилиндры модели 4KS относятся к цилиндрам класса Hi-End Security и являются одними из самых передовых и защищенных от всех методов вскрытия механизмами. 4KS пришел на смену легендарного по своим характеристикам и комфорту пользования цилиндру 3KS Plus. Данные цилиндры обладают абсолютной стойкостью к отмычкам и не вскрываются с помощью ударной техники - бампинга, обладают высочайшими показателями механической прочности и долговечности. ОСОБЕННОСТИ МЕХАНИЗМА 12 НЕЗАВИСИМЫХ КОДОВЫХ ЭЛЕМЕНТОВ
Три кодовых паза на ключе перемещают 12 независимых друг от друга кодовых элементов («слайдеров») внутри цилиндра. При этом блокировка осуществляется не самими «слайдерами», а двумя запорными планками, фиксирующими вращающийся элемент («плаг») в корпусе по всей длине. Кроме того, торцы ключа имеет нарезку, распознающуюся специальным элементом внутри цилиндра. УНИКАЛЬНОСТЬ КАЖДОГО КЛЮЧА СИСТЕМЫ 4KS
Компания EVVA гарантирует уникальность каждого ключа системы 4KS. По заявлениям австрийского производителя, в системе 4KS возможно существование более 30.000.000.000(тридцать биллионов) уникальныx двусторонних ключей. Ключ 4KS имеет неординарную форму, является реверсивным, что позволяет вставлять его любой стороной. Каждый ключ уникален и неповторим, как отпечаток пальца. Изготавливается ключ из мельхиора(низельбер) – износостойкий материал, обеспечивающий огромный ресурс работы и минимальный износ как самого лезвия ключа, так и всего механизма в целом. ЗАЩИТА ОТ ВСКРЫТИЯ И ВЗЛОМА
ЗАЩИТА ОТ НЕСАНКЦИОНИРОВАННОГО КОПИРОВАНИЯ КЛЮЧЕЙ Болванки ключей 3KS защищены международным патентам до 2036 года и распространяются только среди партнеров EVVA. Изготовление копии ключа возможно только при предъявлении карточки владельца и только в авторизованных центрах. Специалисты silent.by будут рады помочь с изготовлением утерянного или дополнительного ключа EVVA, для изготовления уникального ключа, наличие карточки обязательно! ПОВЫШЕННАЯ НАДЕЖНОСТЬ
Благодаря отсутствию пружин и узких каналов цилиндровые механизмы системы 4КS отличаются повышенной стойкостью к засорению и обмерзанию и могут использоваться в неблагоприятных условиях. Длинная шейка ключей позволяет использовать их в сочетании практически с любыми дополнительными защитами цилиндров, присутствующими на рынке. УДОБСТВО ИСПОЛЬЗОВАНИЯ
При разработке системы цилиндров EVVA 4КS были произведены улучшения ключа по сравнению с ЗКS: острые края ключа были несколько скруглены, а уменьшение глубины фрезеровки повысило прочность и стабилность работы ключа. Цилиндры 4КS могут выполняться с ключами имеющими разноцветные пластиковые головки (на выбор: красный, синий, желтый, зеленый, черный), а так же разноцветными насадками для различия между ключами членов семьи, работников офиса или в простых и сложных мастер-системах.
Если Вы ищете интересный и эксклюзивный подарок для близкого Вам человека, друга, коллеги, партнера, то для этого отлично подойдет эксклюзивная кружка,чашка с фотографией либо же с изображением его хобби, фирменного логотипа, либо же любого другого изображения, все ограничивается только Вашей фантазией.
Так же у нас всегда есть чашки в ассортименте и мы своевременно закупаем новые модели сублимационных чашек. Клиенту всегда есть из чего выбрать. Тарелка с фотографией - прекрасный подарок, особенно он ценен для родителей, бабушек и дедушек. Тарелки с декором комплектуются подставками и безукоризненно смотрятся на кухонной полке в качестве декоративного панно. К тому же, из них можно есть! :) А тарелочки с изображением любимых героев мультфильмов очень нравятся детям.
Сувенирная тарелка с видами города - станет практичным сувениром для гостей города, партнеров из других городов. Тарелки с изображением пользуются спросом в качестве подарков для конкурсов при проведении тематических вечеринок в клубах и во время праздничных мероприятий. Праздничные тарелки фотографией молодоженов - станет неповторимым подарком на Свадьбу, сувениром для гостей или удивят присутствующих, являясь оригинальным элементом сервировки праздничного банкета.""",
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
            "filtered": 191,
            "kept": 461,
            "visited": 652,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 34,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 0,
        },
        "final_lines": {"description": "How many lines were actually written.", "value": 200},
    },
    "ru": {
        "shards": {
            "description": "How many shards were sampled from. Each shard contains a subset of the total datasets available.",
            "filtered": 0,
            "kept": 2,
            "visited": 2,
        },
        "visited_lines": {
            "description": "How many lines were visited and kept from the HPLT documents.",
            "filtered": 193,
            "kept": 621,
            "visited": 814,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 102,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 93,
        },
        "final_lines": {"description": "How many lines were actually written.", "value": 200},
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


def config(trg_lang):
    zh_config_path = os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.enzh.yml"))
    return zh_config_path if trg_lang == "zh" else None


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_data_importer")


@pytest.mark.parametrize(
    "importer,trg_lang,dataset",
    [
        ("mtdata", "ru", "Neulab-tedtalks_test-1-eng-rus"),
        ("opus", "ru", "ELRC-3075-wikipedia_health_v1"),
        ("flores", "ru", "dev"),
        ("flores", "zh", "dev"),
        ("sacrebleu", "ru", "wmt19"),
        ("url", "ru", "gcp_pytest-dataset_a0017e"),
    ],
)
def test_basic_corpus_import(importer, trg_lang, dataset, data_dir):
    data_dir.run_task(
        f"dataset-{importer}-{dataset}-en-{trg_lang}",
        env={
            "WGET": os.path.join(CURRENT_FOLDER, "fixtures/wget"),
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
        config=config(trg_lang),
    )

    prefix = data_dir.join(f"artifacts/{dataset}")
    output_src = f"{prefix}.en.zst"
    output_trg = f"{prefix}.{trg_lang}.zst"

    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    assert len(read_lines(output_src)) > 0
    assert len(read_lines(output_trg)) > 0


mono_params = [
    ("news-crawl", "en", "news_2021",                    [0, 1, 4, 6, 3, 7, 5, 2]),
    ("news-crawl", "ru", "news_2021",                    [0, 1, 4, 6, 3, 7, 5, 2]),
    ("news-crawl", "zh", "news_2021",                    [0, 1, 4, 6, 3, 7, 5, 2]),
    ("url",        "en", "gcp_pytest-dataset_en_cdd0d7", [2, 1, 5, 4, 0, 7, 6, 3]),
    ("url",        "ru", "gcp_pytest-dataset_ru_be3263", [5, 4, 2, 0, 7, 1, 3, 6]),
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
        config=config(language),
    )

    prefix = data_dir.join(f"artifacts/{dataset}")
    mono_data = f"{prefix}.{language}.zst"

    data_dir.print_tree()

    sample = {"en": en_sample, "ru": ru_sample, "zh": zh_sample}

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
    dataset = "mono_v2_0"
    data_dir.print_tree()
    max_sentences = 200
    max_characters = 600

    data_dir.run_task(
        f"dataset-hplt-{dataset}-{language}",
        env={
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
        extra_args=[
            "--max_sentences",
            str(max_sentences),
            "--hplt_max_characters",
            str(max_characters),
        ],
    )
    data_dir.print_tree()

    lines = read_lines(data_dir.join(f"artifacts/{dataset}.{language}.zst"))
    max_len = max(len(l[:-1]) for l in lines)
    assert len(lines) == max_sentences
    assert max_len <= max_characters
    assert max_len > max_characters - 50
    assert (
        json.loads(data_dir.read_text(f"artifacts/{dataset}.{language}.stats.json"))
        == hplt_stats[language]
    )
    assert [l[:-1] for l in lines[:10]] == hplt_expected[language].split("\n")


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


@pytest.mark.parametrize("params", [("ru", "aug-mix"), ("zh", "aug-mix-cjk")])
def test_augmentation_mix(data_dir, params):
    src_lang, modifier = params
    dataset = f"sacrebleu_{modifier}_wmt19"
    original_dataset = "sacrebleu_wmt19"
    prefix = data_dir.join(dataset)
    prefix_original = data_dir.join(original_dataset)
    output_src = f"{prefix}.{src_lang}.zst"
    output_trg = f"{prefix}.{TRG}.zst"
    original_src = f"{prefix_original}.{src_lang}.zst"
    original_trg = f"{prefix_original}.{TRG}.zst"
    run_import("corpus", original_dataset, prefix_original, src=src_lang, trg=TRG)

    run_import("corpus", dataset, prefix, src=src_lang, trg=TRG)

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
