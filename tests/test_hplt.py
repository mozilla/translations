import json

import pytest
import zstandard as zstd

from fixtures import DataDir, get_mocked_downloads

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
ЗАПАТЕНТОВАННЫЙ МЕХАНИЗМ Цилиндры модели 4KS относятся к цилиндрам класса Hi-End Security и являются одними из самых передовых и защищенных от всех методов вскрытия механизмами. 4KS пришел на смену легендарного по своим характеристикам и комфорту пользования цилиндру 3KS Plus. Данные цилиндры обладают абсолютной стойкостью к отмычкам и не вскрываются с помощью ударной техники - бампинга, обладают высочайшими показателями механической прочности и долговечности. ОСОБЕННОСТИ МЕХАНИЗМА
12 НЕЗАВИСИМЫХ КОДОВЫХ ЭЛЕМЕНТОВ Три кодовых паза на ключе перемещают 12 независимых друг от друга кодовых элементов («слайдеров») внутри цилиндра. При этом блокировка осуществляется не самими «слайдерами», а двумя запорными планками, фиксирующими вращающийся элемент («плаг») в корпусе по всей длине. Кроме того, торцы ключа имеет нарезку, распознающуюся специальным элементом внутри цилиндра. УНИКАЛЬНОСТЬ КАЖДОГО КЛЮЧА СИСТЕМЫ 4KS
Компания EVVA гарантирует уникальность каждого ключа системы 4KS. По заявлениям австрийского производителя, в системе 4KS возможно существование более 30.000.000.000(тридцать биллионов) уникальныx двусторонних ключей.
Ключ 4KS имеет неординарную форму, является реверсивным, что позволяет вставлять его любой стороной. Каждый ключ уникален и неповторим, как отпечаток пальца. Изготавливается ключ из мельхиора(низельбер) – износостойкий материал, обеспечивающий огромный ресурс работы и минимальный износ как самого лезвия ключа, так и всего механизма в целом. ЗАЩИТА ОТ ВСКРЫТИЯ И ВЗЛОМА
ЗАЩИТА ОТ НЕСАНКЦИОНИРОВАННОГО КОПИРОВАНИЯ КЛЮЧЕЙ Болванки ключей 3KS защищены международным патентам до 2036 года и распространяются только среди партнеров EVVA. Изготовление копии ключа возможно только при предъявлении карточки владельца и только в авторизованных центрах. Специалисты silent.by будут рады помочь с изготовлением утерянного или дополнительного ключа EVVA, для изготовления уникального ключа, наличие карточки обязательно! ПОВЫШЕННАЯ НАДЕЖНОСТЬ
Благодаря отсутствию пружин и узких каналов цилиндровые механизмы системы 4КS отличаются повышенной стойкостью к засорению и обмерзанию и могут использоваться в неблагоприятных условиях. Длинная шейка ключей позволяет использовать их в сочетании практически с любыми дополнительными защитами цилиндров, присутствующими на рынке. УДОБСТВО ИСПОЛЬЗОВАНИЯ
При разработке системы цилиндров EVVA 4КS были произведены улучшения ключа по сравнению с ЗКS: острые края ключа были несколько скруглены, а уменьшение глубины фрезеровки повысило прочность и стабилность работы ключа. Цилиндры 4КS могут выполняться с ключами имеющими разноцветные пластиковые головки (на выбор: красный, синий, желтый, зеленый, черный), а так же разноцветными насадками для различия между ключами членов семьи, работников офиса или в простых и сложных мастер-системах.
Если Вы ищете интересный и эксклюзивный подарок для близкого Вам человека, друга, коллеги, партнера, то для этого отлично подойдет эксклюзивная кружка,чашка с фотографией либо же с изображением его хобби, фирменного логотипа, либо же любого другого изображения, все ограничивается только Вашей фантазией.
Так же у нас всегда есть чашки в ассортименте и мы своевременно закупаем новые модели сублимационных чашек. Клиенту всегда есть из чего выбрать. Тарелка с фотографией - прекрасный подарок, особенно он ценен для родителей, бабушек и дедушек. Тарелки с декором комплектуются подставками и безукоризненно смотрятся на кухонной полке в качестве декоративного панно. К тому же, из них можно есть! :) А тарелочки с изображением любимых героев мультфильмов очень нравятся детям.""",
}

hplt_stats = {
    "en": {
        "shards": {
            "description": "How many shards were sampled from. Each shard contains a subset of the total datasets available.",
            "filtered": 0,
            "kept": 2,
            "visited": 2,
        },
        "visited_lines": {
            "description": "How many lines were visited and kept from the HPLT documents.",
            "filtered": 371,
            "kept": 1197,
            "visited": 1568,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 103,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 48,
        },
        "final_lines": {"description": "How many lines were actually written.", "value": 500},
        "filtered_doc_locale": {
            "description": "How many lines were filtered based on document locale.",
            "value": 0,
        },
        "filtered_line_locale": {
            "description": "How many lines were filtered based on line locales.",
            "value": 218,
        },
        "filtered_doc_score": {
            "description": "How many lines were filtered based on document scores.",
            "value": 28,
        },
        "filtered_too_long": {
            "description": "How many lines were filtered based on length.",
            "value": 153,
        },
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
            "filtered": 209,
            "kept": 566,
            "visited": 775,
        },
        "document_count": {
            "description": "How many documents were visited. This can help represent data diversity.",
            "value": 86,
        },
        "duplicate_lines": {
            "description": "Of the collected lines, this counts how many were duplicates and discarded.",
            "value": 96,
        },
        "final_lines": {"description": "How many lines were actually written.", "value": 200},
        "filtered_doc_locale": {
            "description": "How many lines were filtered based on document locale.",
            "value": 0,
        },
        "filtered_line_locale": {
            "description": "How many lines were filtered based on line locales.",
            "value": 62,
        },
        "filtered_doc_score": {
            "description": "How many lines were filtered based on document scores.",
            "value": 69,
        },
        "filtered_too_long": {
            "description": "How many lines were filtered based on length.",
            "value": 147,
        },
    },
}


def read_lines(path):
    with zstd.open(path, "rt") as f:
        return f.readlines()


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_hplt")


def test_mono_hplt_merging(data_dir: DataDir):
    """
    Test mono HPLT downloading with segments accumulation mode
    """
    dataset = "mono_v3_0"
    language = "en"
    data_dir.print_tree()
    max_sentences = 500
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
            "--hplt_merge_lines",
            "True",
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


def test_mono_hplt_no_merging(data_dir: DataDir):
    """
    Test mono HPLT downloading without segments accumulation mode
    """
    dataset = "mono_v3_0"
    language = "ru"
    data_dir.print_tree()
    max_sentences = 200
    max_characters = 500

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
            "--hplt_merge_lines",
            "False",
        ],
    )
    data_dir.print_tree()

    lines = read_lines(data_dir.join(f"artifacts/{dataset}.{language}.zst"))
    max_len = max(len(l[:-1]) for l in lines)
    assert len(lines) == max_sentences
    assert max_len <= max_characters
    assert (
        json.loads(data_dir.read_text(f"artifacts/{dataset}.{language}.stats.json"))
        == hplt_stats[language]
    )
    assert [l[:-1] for l in lines[:10]] == hplt_expected[language].split("\n")
