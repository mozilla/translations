import shutil
import pytest
from pathlib import Path
from fixtures import DataDir
from pipeline.common.downloads import stream_download_to_file
from typing import Union


text = """La màfia no va recuperar el seu poder fins al cap de la rendició d'Itàlia en la Segona Guerra Mundial.
En els vuitanta i noranta, una sèrie de disputes internes van portar a la mort a molts membres destacats de la màfia.
Després del final de la Segona Guerra Mundial, la màfia es va convertir en un Estat dins de l'Estat.
Els seus tentacles ja no abastaven només a Sicília, sinó gairebé a tota l'estructura econòmica d'Itàlia, i d'usar escopetes de canons retallats, va passar a disposar d'armament més expeditiu: revòlvers del calibre .357 Magnum, fusells llança-granades, bazookas, i explosius.
La màfia i altres societats secretes del crim organitzat van formar un sistema de vasos comunicants.
En la lògia maçònica P-2, representada pel gran maestre Lici Gelli, hi havia ministres, parlamentaris, generals, jutges, policies, banquers, aristòcrates i fins i tot mafiosos.
En 1992, la màfia siciliana va assassinar al jutge italià Giovanni Falcone fent esclatar mil quilograms d'explosius col·locats sota l'autopista que uneix Palerm amb l'aeroport ara anomenat Giovanni Falcone.
Van morir ell, la seva esposa Francesca Morvilio i tres escortes.
En 1993, cinc ex-presidents de Govern, moltíssims ministres i més de 3000 polítics i empresaris van ser acusats, processaments o condemnats per corrupció i associació amb la màfia.
Es tractava d'un missatge de la màfia al vell Andreotti, ex-president del Govern, per no aturar l'enpresonament masiu dels seus membres.
La màfia no perdona mai, com ja no podran testificar els banquers Michele Sindona i Roberto Calvi, dos mags de les finances del Vaticà, la màfia i altres institucions d'Itàlia.
Van ser assassinats per un rampell de cobdícia, ja que van voler apropiar-se dels diners de la màfia.
El capo di tutti capi és el major rang que pot haver-hi en la Cosa Nostra.
Es tracta del cap d'una família que, en ser més poderós o per haver assassinat als altres caps de les altres famílies, s'ha convertit en el més poderós membre de la màfia.
Un exemple d'això va ser Salvatore Maranzano , qui va ser traït per Lucky Luciano, qui finalment li va cedir el lloc ―en ser extradit per problemes amb la justícia nord-americana― a la seva mà dreta i conseller, Frank Costello.
El don és el cap d'una família.
"""
text2 = """माफिया ने द्वितीय विश्व युद्ध में इटली के आत्मसमर्पण के बाद ही अपनी शक्ति वापस पाई।
अस्सी और नब्बे के दशक में, आंतरिक विवादों की एक श्रृंखला के कारण माफिया के कई प्रमुख सदस्यों की मौत हो गई।
द्वितीय विश्व युद्ध की समाप्ति के बाद, माफिया राज्य के भीतर एक राज्य बन गया।
इसके जाल अब केवल सिसिली तक ही सीमित नहीं थे, बल्कि इटली के लगभग पूरे आर्थिक ढांचे तक फैल चुके थे, और कटी हुई नली वाली बंदूकें इस्तेमाल करने से लेकर, यह अधिक मारक हथियारों तक पहुँच गया: .357 मैग्नम कैलिबर रिवॉल्वर, ग्रेनेड लॉन्चर राइफलें, बाज़ूका और विस्फोटक।
माफिया और संगठित अपराध के अन्य गुप्त समाजों ने आपस में जुड़े हुए तंत्र की एक प्रणाली बनाई।
ग्रैंड मास्टर लिसियो गेली द्वारा प्रतिनिधित्व किए गए P-2 मेसोनिक लॉज में मंत्री, सांसद, जनरल, न्यायाधीश, पुलिस अधिकारी, बैंकर, कुलीन और यहाँ तक कि माफिया भी शामिल थे।
1992 में, सिसिली के माफिया ने इतालवी न्यायाधीश जियोवानी फाल्कोन की हत्या कर दी, जिसके लिए पलेर्मो को अब जियोवानी फाल्कोन नामक हवाई अड्डे से जोड़ने वाले राजमार्ग के नीचे रखे गए एक हज़ार किलोग्राम विस्फोटकों में धमाका किया गया था।
इसमें उनकी, उनकी पत्नी फ्रांसेस्का मोर्विलो और तीन अंगरक्षकों की मौत हो गई।
1993 में, सरकार के पांच पूर्व प्रमुखों, कई मंत्रियों और 3000 से अधिक राजनेताओं और व्यापारियों पर भ्रष्टाचार और माफिया के साथ संबंध होने का आरोप लगाया गया, मुकदमा चलाया गया या उन्हें दोषी ठहराया गया।
यह माफिया की ओर से सरकार के पूर्व प्रमुख, बुजुर्ग आंद्रेओटी के लिए एक संदेश था, क्योंकि उन्होंने इसके सदस्यों को सामूहिक रूप से जेल में डाले जाने से नहीं रोका था।
माफिया कभी माफ नहीं करता, जैसा कि बैंकर मिशेल सिंडोना और रॉबर्टो काल्वी अब गवाही नहीं दे पाएंगे, जो वेटिकन, माफिया और इटली के अन्य संस्थानों के वित्त के दो जादूगर थे।
लालच के आवेश में उनकी हत्या कर दी गई, क्योंकि वे माफिया के पैसे हड़पना चाहते थे।
'कापो डी टुटी कापी' कोसा नोस्ट्रा में सबसे बड़ा पद होता है।
यह एक ऐसे परिवार का मुखिया होता है जो अधिक शक्तिशाली होने के कारण या अन्य परिवारों के अन्य मुखियों की हत्या करने के बाद, माफिया का सबसे शक्तिशाली सदस्य बन गया है।
इसका एक उदाहरण साल्वातोरे मारनज़ानो था, जिसे लकी लूसियानो ने धोखा दिया था, जिसने अंततः अमेरिकी न्याय प्रणाली के साथ समस्याओं के कारण प्रत्यर्पित किए जाने पर, अपना स्थान अपने दाहिने हाथ और सलाहकार, फ्रैंक कॉस्टेलो को सौंप दिया।
'डॉन' एक परिवार का मुखिया होता है।
"""


@pytest.fixture
def data_dir():
    data_dir = DataDir("test_ctranslate2")
    return data_dir


def download_and_cache(data_dir: DataDir, url: str, cached_filename: str, data_dir_name: str):
    """
    Download remote language model resources and cache them in the data directory.
    """
    src_dir = Path(__file__).parent.parent
    cached_file = src_dir / "data/tests" / cached_filename
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    if not cached_file.exists():
        stream_download_to_file(url, cached_file)
    shutil.copy(cached_file, data_dir.join(data_dir_name))


@pytest.mark.parametrize(
    "input_,expected_output,decoder,extra_flags,extra_args",
    [
        (
            text,
            [
                "The Mafia did not regain its power until the end of World War II.",
                "In the 1990s, a series of internal scandals led to the death of many prominent members of the Mafia.",
                "After World War II, the Mafia became a state.",
                "The Italians, however, did not concentrate more heavily on the use of steel, but only in the case of the most expensive and costly weaponry: the Babylonian cartridges, with more than 3,500 rifles, were eroded, all of them eroded, including the cryptanalysts, the cylindrical rifles, the cylindrical rifles and the cylindrical rifles.",
                "The Mafia and other secret societies formed a system of organized crime.",
                "In the Ptolemy II, the Giulio Giulio Giulio Giulio, which included a number of magistrates, magistrates, magistrates, magistrates, magistrates, magistrates, magistrates, magistrates, ministers, even the police, even the police, the police, the police, the police, the police, the police, the police and the police.",
                "In 1992, the Italian dictator Giovanni Falcone shot down the 600-year-old paratrooper Giovanni Falcone with a parachute carrying a paratrooper named Giovanni Falcone.",
                "He was succeeded by his wife, Francesco Morgas, and three sisters.",
                "In 1993, more than three hundred ministers, lawyers and government officials were accused and accused of fraud and corruption.",
                "It was a message from Mr Andreotti, the former prime minister, to stop the government's membership.",
                "The bankers will not be able to imagine, as Michel Salmond and Pablo Guerrero have said, the bankers of the two banks of the Vatican and the Vatican.",
                "They were murdered by a mafia because they wanted to steal money from the mafia.",
                "The Capricorn is the largest rank that can be in our heads.",
                "It is the head of a family, or more powerful than the head of another family, who has become the most powerful Mafia leader.",
                "This was a tragedy that Luciano Margo, who was later persuaded by Frank Prigogore, gave up for Lucca, who was incarcerated by Luca Cortino, who was later incarcerated by Margo.",
                "The head is not the head of a family.",
            ],
            "ctranslate2",
            None,
            None,
        ),
        (
            text,
            [
                "The Mafia was losing its capacity to surrender to Egypt during the Allied invasion of World War II.",
                "In the late 1990s, a series of internal scandals created a huge amount of memory among members of the Mafia.",
                "After the end of World War II, the Mafia became a state.",
                "Its operations were still not exclusively of the Italian type, but instead included the most powerful weaponry, the more than 30,000 tons of silver, and the more powerful, but, by using the slender, more effective, the more modern, and more expensive, steel-making rifle, the more sterile, and all of the more expensive, but still more complex rifles: the slender, and the tonna-gun guns.",
                "The Genocide and other secret criminal organisations formed a means of organised crime.",
                "The G7, which included the two major magistrates, Micaio, Mons, Ptolemy, Legona, Geria, and the magistrates including, even by chance, police, minister and magistrat, even magistrates, were a number of police officers, ministers, judges, officials.",
                "In 1992, the Italian dictator Giovanni Falcone killed in a bombing by the paratrooper Giovanni Falcono who, along the thieves' plane, landed 900 people in Palermo under the name of Falcon Falcone Airport.",
                "Morphy, his wife Francesca Moroni, and his 3 assistants were arrested.",
                "In 1997 more than eight government ministers and three hundred political, political and government advisers were detained by members of the police in 1996 and 1997, accused of corruption and drug abuse.",
                "It was a message from the former minister of the Interior, Andrea Bonino, to keep his government' s leader behind closed.",
                "The bailouts will not wait for Mr. Messias and Miguel Pablo Garrios, Banks Secretary Salmon Michel Romero and two other financial advisors like Pablo Savioni, the Vatican, and the media.",
                "They were being kidnapped by a mob because they wanted to murder the money from the mob.",
                "Dharma di Cap is the largest number of organ that is in the head.",
                "It is the father who, being the most powerful leader or responsible for killing the other family members, has become the strongest leader of the family to have been the richest Mossos of the Mafia.",
                "This was the tragedy that Lucino Margono gave to Frank Prigo, who later escaped justice through the United Kingdom, was the subject of controversy, and was also a concern for Lucca.",
                "“Death isn’t a head of a family.",
            ],
            "ctranslate2",
            None,
            ["--beam-size", "1", "--output-sampling", "[topk,", "10]"],
        ),
        (
            text2,
            [
                "The Mafia only regained its power after Italy's surrender in World War II.",
                "In the eighties and nineties, a series of internal disputes led to the deaths of several prominent members of the Mafia.",
                "After the end of World War II, the Mafia became a state within a state.",
                "Its traps were no longer confined to Sicily, but extended to almost the entire economic structure of Italy, and from using cut-throat guns, it reached more lethal weapons: .357 Magnum caliber revolvers, grenade launcher rifles, bazookas, and explosives.",
                "The Mafia and other secret societies of organized crime created a system of interconnected mechanisms.",
                "The P-2 Masonic Lodge, represented by Grand Master Licio Gelli, included ministers, parliamentarians, generals, judges, police officers, bankers, nobles, and even the Mafia.",
                "In 1992, the Sicilian Mafia assassinated Italian judge Giovanni Falcone, by detonating a thousand kilograms of explosives placed under the highway connecting Palermo to the airport now called Giovanni Falcone.",
                "He, his wife Francesca Morvillo, and three bodyguards were killed.",
                "In 1993, five former heads of government, several ministers, and over 3000 politicians and businessmen were charged, tried, or convicted of corruption and links with the mafia.",
                "This was a message from the Mafia to the elderly Andreotti, the former head of government, as he had not prevented its members from being jailed en masse.",
                "The Mafia never forgives, as the bankers Michele Sindona and Roberto Calvi will no longer be able to testify, two wizards of the finances of the Vatican, the Mafia and other institutions in Italy.",
                "He was killed out of greed, as he wanted to grab the mafia's money.",
                'The "capo de tutti capi" is the highest rank in Cosa Nostra.',
                "It is the head of a family who, by virtue of being more powerful or having murdered other heads of other families, has become the most powerful member of the Mafia.",
                "An example of this was Salvatore Maranzano, who was betrayed by Lucky Luciano, who eventually ceded his position to his right-hand man and mentor, Frank Costello, when he was extradited due to problems with the US justice system.",
                "The 'Don' is the head of a family.",
            ],
            "indictrans2",
            ["--src_locale", "hi", "--trg_locale", "en"],
            None,
        ),
    ],
    ids=["translate", "translate-topk10", "translate-indictrans2"],
)
def test_ctranslate2(
    input_: str,
    expected_output: list[str],
    decoder: str,
    extra_flags: Union[list[str], None],
    extra_args: Union[list[str], None],
):
    data_dir = DataDir("test_ctranslate2")
    data_dir.mkdir("model1")
    data_dir.create_zst("file.1.zst", input_)

    # Download the teacher models.
    download_and_cache(
        data_dir,
        "https://storage.googleapis.com/releng-translations-dev/models/ca-en/dev/teacher-finetuned1/final.model.npz.best-chrf.npz",
        cached_filename="en-ca-teacher-1.npz",
        data_dir_name="model1/final.model.npz.best-chrf.npz",
    )

    # Download the vocab.
    download_and_cache(
        data_dir,
        "https://storage.googleapis.com/releng-translations-dev/models/ca-en/dev/vocab/vocab.spm",
        cached_filename="en-ca-vocab.spm",
        data_dir_name="vocab.en.spm",
    )
    shutil.copyfile(data_dir.join("vocab.en.spm"), data_dir.join("vocab.ru.spm"))

    data_dir.run_task(
        "distillation-mono-src-translate-en-ru-1/10",
        env={"USE_CPU": "true"},
        # Applied before the "--"
        extra_flags=["--decoder", decoder, "--device", "cpu", *(extra_flags or ())],
        extra_args=extra_args,
    )
    data_dir.print_tree()

    out_lines = data_dir.read_text("artifacts/file.1.out.zst").strip().split("\n")
    assert out_lines == expected_output
