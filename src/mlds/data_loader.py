import os
import json
from enum import Enum
from os.path import join as pj
from typing import List

from pydantic import BaseModel

# fmt: off
LANGUAGES = [
    "amh", "ewe", "hau", "ibo", "kin", 
    "lin", "lug", "orm", "sna", "sot", 
    "swa", "twi", "wol", "xho", "yor", "zul"
]

LANGUAGES_MAPPER = {
    "amh": "Amharic",
    "ewe": "Ewe",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "sna": "Shona",
    "sot": "Sotho",
    "swa": "Swahili",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
    "eng": "English",
}


SLOTS_MAPPER = {
    "LANGUAGE_NAME": 1,
    "CITY_NAME": 2,
    "PLUG_TYPE": 3,
    "COUNTRY": 4,
    "STATE_OR_PROVINCE": 5,
    "CONTINENT": 6,
    "NATIONALITY": 7,
    "PLACE_NAME": 8,
    "ACCOUNT_TYPE": 9,
    "MONEY": 0,
    "CURRENCY": "q",
    "BANK_NAME": "w",
    "PAYMENT_COMPANY": "e",
    "BILL_TYPE": "t",
    "SHOPPING_ITEM": "a",
    "FOOD_ITEM": "s",
    "RESTAURANT_NAME": "d",
    "DISH_NAME": "f",
    "MEAL_PERIOD": "g",
    "SUPERMARKET_NAME": "z",
    "TIME": "x",
    "DATE": "c",
    "HOTEL_NAME": "v",
    "CALENDAR_EVENT": "b",
    "AIRPORT_NAME": "y",
    "AIRLINE": "i",
    "CAR_TYPE": "o",
    "TIMEZONE": "p",
    "NUMBER": "j",
    "PERSONAL_NAME": "k",
    "CAR_RENTAL_COMPANY": "l",
    "MUSIC_GENRE": "n",
    "ARTIST_NAME": "m",
    "SONG_NAME": "u",  # u id define by myself
}

SLOTS = list(sorted(SLOTS_MAPPER.keys()))

INTENTS = [
    "alarm",
    "balance",
    "bill_balance",
    "book_flight",
    "book_hotel",
    "calendar_update",
    "cancel_reservation",
    "car_rental",
    "confirm_reservation",
    "cook_time",
    "exchange_rate",
    "food_last",
    "freeze_account",
    "ingredients_list",
    "interest_rate",
    "international_visa",
    "make_call",
    "meal_suggestion",
    "min_payment",
    "pay_bill",
    "pin_change",
    "play_music",
    "plug_type",
    "recipe",
    "restaurant_reservation",
    "restaurant_reviews",
    "restaurant_suggestion",
    "share_location",
    "shopping_list_update",
    "spending_history",
    "text",
    "time",
    "timezone",
    "transactions",
    "transfer",
    "translate",
    "travel_notification",
    "travel_suggestion",
    "update_playlist",
    "weather",
]


class SlotEnum(str, Enum):
    LANGUAGE_NAME = "LANGUAGE_NAME"
    CITY_NAME = "CITY_NAME"
    PLUG_TYPE = "PLUG_TYPE"
    COUNTRY = "COUNTRY"
    STATE_OR_PROVINCE = "STATE_OR_PROVINCE"
    CONTINENT = "CONTINENT"
    NATIONALITY = "NATIONALITY"
    PLACE_NAME = "PLACE_NAME"
    ACCOUNT_TYPE = "ACCOUNT_TYPE"
    MONEY = "MONEY"
    CURRENCY = "CURRENCY"
    BANK_NAME = "BANK_NAME"
    PAYMENT_COMPANY = "PAYMENT_COMPANY"
    BILL_TYPE = "BILL_TYPE"
    SHOPPING_ITEM = "SHOPPING_ITEM"
    FOOD_ITEM = "FOOD_ITEM"
    RESTAURANT_NAME = "RESTAURANT_NAME"
    DISH_NAME = "DISH_NAME"
    MEAL_PERIOD = "MEAL_PERIOD"
    SUPERMARKET_NAME = "SUPERMARKET_NAME"
    TIME = "TIME"
    DATE = "DATE"
    HOTEL_NAME = "HOTEL_NAME"
    CALENDAR_EVENT = "CALENDAR_EVENT"
    AIRPORT_NAME = "AIRPORT_NAME"
    AIRLINE = "AIRLINE"
    CAR_TYPE = "CAR_TYPE"
    TIMEZONE = "TIMEZONE"
    NUMBER = "NUMBER"
    PERSONAL_NAME = "PERSONAL_NAME"
    CAR_RENTAL_COMPANY = "CAR_RENTAL_COMPANY"
    MUSIC_GENRE = "MUSIC_GENRE"
    ARTIST_NAME = "ARTIST_NAME"
    SONG_NAME = "SONG_NAME"


class IntentsEnum(str, Enum):
    ALARM = "alarm"
    BALANCE = "balance"
    BILL_BALANCE = "bill_balance"
    BOOK_FLIGHT = "book_flight"
    BOOK_HOTEL = "book_hotel"
    CALENDAR_UPDATE = "calendar_update"
    CANCEL_RESERVATION = "cancel_reservation"
    CAR_RENTAL = "car_rental"
    CONFIRM_RESERVATION = "confirm_reservation"
    COOK_TIME = "cook_time"
    EXCHANGE_RATE = "exchange_rate"
    FOOD_LAST = "food_last"
    FREEZE_ACCOUNT = "freeze_account"
    INGREDIENTS_LIST = "ingredients_list"
    INTEREST_RATE = "interest_rate"
    INTERNATIONAL_VISA = "international_visa"
    MAKE_CALL = "make_call"
    MEAL_SUGGESTION = "meal_suggestion"
    MIN_PAYMENT = "min_payment"
    PAY_BILL = "pay_bill"
    PIN_CHANGE = "pin_change"
    PLAY_MUSIC = "play_music"
    PLUG_TYPE = "plug_type"
    RECIPE = "recipe"
    RESTAURANT_RESERVATION = "restaurant_reservation"
    RESTAURANT_REVIEWS = "restaurant_reviews"
    RESTAURANT_SUGGESTION = "restaurant_suggestion"
    SHARE_LOCATION = "share_location"
    SHOPPING_LIST_UPDATE = "shopping_list_update"
    SPENDING_HISTORY = "spending_history"
    TEXT = "text"
    TIME = "time"
    TIMEZONE = "timezone"
    TRANSACTIONS = "transactions"
    TRANSFER = "transfer"
    TRANSLATE = "translate"
    TRAVEL_NOTIFICATION = "travel_notification"
    TRAVEL_SUGGESTION = "travel_suggestion"
    UPDATE_PLAYLIST = "update_playlist"
    WEATHER = "weather"


class Slot(BaseModel):
    slot: SlotEnum
    entity: str


class IntentAndSlots(BaseModel):
    intention: IntentsEnum
    slots: List[Slot]


SCHEMA = IntentAndSlots.model_json_schema()

# fmt: on


# fmt: off
SLOTS_MERGED_MAPPER = {
    "LANGUAGE_NAME": 1,
    "CITY_OR_PROVINCE": 2, # CITY_NAME
    # "PLUG_TYPE": 3,
    "COUNTRY": 4,
    # "STATE_OR_PROVINCE": 5,
    # "CONTINENT": 6,
    # "NATIONALITY": 7,
    "PLACE_NAME": 8,
    "ACCOUNT_TYPE": 9,
    "MONEY": 0,
    "CURRENCY": "q",
    "BANK_NAME": "w",
    "PAYMENT_COMPANY": "e",
    "BILL_TYPE": "t",
    "SHOPPING_ITEM": "a",
    "DISH_OR_FOOD": "s", # FOOD_ITEM
    "RESTAURANT_NAME": "d",
    # "DISH_NAME": "f",
    "MEAL_PERIOD": "g",
    # "SUPERMARKET_NAME": "z",
    "TIME": "x",
    "DATE": "c",
    "HOTEL_NAME": "v",
    "CALENDAR_EVENT": "b",
    # "AIRPORT_NAME": "y",
    # "AIRLINE": "i",
    # "CAR_TYPE": "o",
    # "TIMEZONE": "p",
    "NUMBER": "j",
    "PERSONAL_NAME": "k",
    # "CAR_RENTAL_COMPANY": "l",
    "MUSIC_GENRE": "n",
    "ARTIST_NAME": "m",
    "SONG_NAME": "u",  # u id define by myself
}

MERGE_MAPPER = {
    # "DISH_NAME": "FOOD_ITEM",
    # "STATE_OR_PROVINCE": "CITY_NAME"
    "DISH_NAME": "DISH_OR_FOOD",
    "FOOD_ITEM": "DISH_OR_FOOD",
    "STATE_OR_PROVINCE": "CITY_OR_PROVINCE",
    "CITY_NAME": "CITY_OR_PROVINCE",
    # CITY_OR_PROVINCE
}

SLOTS_MERGED = list(sorted(SLOTS_MERGED_MAPPER.keys()))


class SlotMergedEnum(str, Enum):
    LANGUAGE_NAME = "LANGUAGE_NAME"
    CITY_NAME = "CITY_NAME"
    COUNTRY = "COUNTRY"
    PLACE_NAME = "PLACE_NAME"
    ACCOUNT_TYPE = "ACCOUNT_TYPE"
    MONEY = "MONEY"
    CURRENCY = "CURRENCY"
    BANK_NAME = "BANK_NAME"
    PAYMENT_COMPANY = "PAYMENT_COMPANY"
    BILL_TYPE = "BILL_TYPE"
    SHOPPING_ITEM = "SHOPPING_ITEM"
    FOOD_ITEM = "FOOD_ITEM"
    RESTAURANT_NAME = "RESTAURANT_NAME"
    MEAL_PERIOD = "MEAL_PERIOD"
    TIME = "TIME"
    DATE = "DATE"
    HOTEL_NAME = "HOTEL_NAME"
    CALENDAR_EVENT = "CALENDAR_EVENT"
    NUMBER = "NUMBER"
    PERSONAL_NAME = "PERSONAL_NAME"
    MUSIC_GENRE = "MUSIC_GENRE"
    ARTIST_NAME = "ARTIST_NAME"
    SONG_NAME = "SONG_NAME"


class SlotMerged(BaseModel):
    slot: SlotMergedEnum
    entity: str


class IntentAndSlots(BaseModel):
    intention: IntentsEnum
    slots: List[SlotMerged]


SCHEMA_MERGED = IntentAndSlots.model_json_schema()

NLLB_MAPPER = {
    "amh": "amh_Ethi",  #
    "eng": "eng_Latn",
    "ewe": "ewe_Latn",
    "hau": "hau_Latn",
    "ibo": "ibo_Latn",
    "kin": "kin_Latn",
    "lin": "lin_Latn",
    "lug": "lug_Latn",
    "orm": "som_Latn",
    "sna": "sna_Latn",
    "sot": "sot_Latn",
    "swa": "swh_Latn",
    "twi": "twi_Latn",
    "wol": "wol_Latn",
    "xho": "xho_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
}


CLINC150_MAPPER = {
    0: "restaurant_reviews",
    1: "nutrition_info",
    2: "account_blocked",
    3: "oil_change_how",
    4: "time",
    5: "weather",
    6: "redeem_rewards",
    7: "interest_rate",
    8: "gas_type",
    9: "accept_reservations",
    10: "smart_home",
    11: "user_name",
    12: "report_lost_card",
    13: "repeat",
    14: "whisper_mode",
    15: "what_are_your_hobbies",
    16: "order",
    17: "jump_start",
    18: "schedule_meeting",
    19: "meeting_schedule",
    20: "freeze_account",
    21: "what_song",
    22: "meaning_of_life",
    23: "restaurant_reservation",
    24: "traffic",
    25: "make_call",
    26: "text",
    27: "bill_balance",
    28: "improve_credit_score",
    29: "change_language",
    30: "no",
    31: "measurement_conversion",
    32: "timer",
    33: "flip_coin",
    34: "do_you_have_pets",
    35: "balance",
    36: "tell_joke",
    37: "last_maintenance",
    38: "exchange_rate",
    39: "uber",
    40: "car_rental",
    41: "credit_limit",
    42: "oos",
    43: "shopping_list",
    44: "expiration_date",
    45: "routing",
    46: "meal_suggestion",
    47: "tire_change",
    48: "todo_list",
    49: "card_declined",
    50: "rewards_balance",
    51: "change_accent",
    52: "vaccines",
    53: "reminder_update",
    54: "food_last",
    55: "change_ai_name",
    56: "bill_due",
    57: "who_do_you_work_for",
    58: "share_location",
    59: "international_visa",
    60: "calendar",
    61: "translate",
    62: "carry_on",
    63: "book_flight",
    64: "insurance_change",
    65: "todo_list_update",
    66: "timezone",
    67: "cancel_reservation",
    68: "transactions",
    69: "credit_score",
    70: "report_fraud",
    71: "spending_history",
    72: "directions",
    73: "spelling",
    74: "insurance",
    75: "what_is_your_name",
    76: "reminder",
    77: "where_are_you_from",
    78: "distance",
    79: "payday",
    80: "flight_status",
    81: "find_phone",
    82: "greeting",
    83: "alarm",
    84: "order_status",
    85: "confirm_reservation",
    86: "cook_time",
    87: "damaged_card",
    88: "reset_settings",
    89: "pin_change",
    90: "replacement_card_duration",
    91: "new_card",
    92: "roll_dice",
    93: "income",
    94: "taxes",
    95: "date",
    96: "who_made_you",
    97: "pto_request",
    98: "tire_pressure",
    99: "how_old_are_you",
    100: "rollover_401k",
    101: "pto_request_status",
    102: "how_busy",
    103: "application_status",
    104: "recipe",
    105: "calendar_update",
    106: "play_music",
    107: "yes",
    108: "direct_deposit",
    109: "credit_limit_change",
    110: "gas",
    111: "pay_bill",
    112: "ingredients_list",
    113: "lost_luggage",
    114: "goodbye",
    115: "what_can_i_ask_you",
    116: "book_hotel",
    117: "are_you_a_bot",
    118: "next_song",
    119: "change_speed",
    120: "plug_type",
    121: "maybe",
    122: "w2",
    123: "oil_change_when",
    124: "thank_you",
    125: "shopping_list_update",
    126: "pto_balance",
    127: "order_checks",
    128: "travel_alert",
    129: "fun_fact",
    130: "sync_device",
    131: "schedule_maintenance",
    132: "apr",
    133: "transfer",
    134: "ingredient_substitution",
    135: "calories",
    136: "current_location",
    137: "international_fees",
    138: "calculator",
    139: "definition",
    140: "next_holiday",
    141: "update_playlist",
    142: "mpg",
    143: "min_payment",
    144: "change_user_name",
    145: "restaurant_suggestion",
    146: "travel_notification",
    147: "cancel",
    148: "pto_used",
    149: "travel_suggestion",
    150: "change_volume",
}
# fmt: on

class EntityDataManager:
    def __init__(self, data_folder="data/json") -> None:
        self.data_folder = data_folder
        # self.languages = ["lin", "orm", "swa", "twi", "hau", "ewe"]
        # languages_files = glob.glob(pj(data_folder, "*.json"))

        # self.data = {}
        # for language in self.languages:
        #     self.data[language] = {}
        #     for reviewed in [True, False]:
        #         fn = self._encode_language(language, reviewed)
        #         self.data[fn] = self._preprocess_json(pj(data_folder, f"{fn}.json"))
        self.eng_cutoff = 120

    def _preprocess_json(self, language, file_path):
        jd = json.load(open(file_path))
        jd.sort(key=lambda x: x["inner_id"])

        # clean strip in entities
        for entry_idx, entry in enumerate(jd):
            text = entry["data"]["text"]
            for anno_idx, annotation in enumerate(entry["annotations"]):
                # annotator_id = annotation["completed_by"]["id"]
                for anno_id, result in enumerate(annotation["result"]):
                    start, end = result["value"]["start"], result["value"]["end"]
                    entity_text:str = text[start:end]
                    if entity_text.lstrip() != entity_text:
                        jd[entry_idx]["annotations"][anno_idx]["result"][anno_id]["value"]["start"] = start + len(entity_text) - len(entity_text.lstrip())
                    if entity_text.rstrip() != entity_text:
                        jd[entry_idx]["annotations"][anno_idx]["result"][anno_id]["value"]["end"] = end - len(entity_text) + len(entity_text.rstrip())
                    jd[entry_idx]["annotations"][anno_idx]["result"][anno_id]["value"]["text"] = result["value"]["text"].strip()

        if language in ["lug", "sna"]:
            eng = jd[:93]
            lang = jd[93:]
            # 3431 - 93 = 3338
            # 3338 - 135 = 3203
        else:
            eng = jd[: self.eng_cutoff]
            lang = jd[self.eng_cutoff :]
        lang = [e for e in lang if len(e["data"]["text"]) >= 3]
        # assert len(lang) == 3200, f"Language data for {file_path} is corrupted"
        return eng, lang

    def _encode_language(self, language: str, reviewed: bool, partial: bool = False):
        return f"{language}_{'reviewed' if reviewed else 'unreviewed'}{'_partially' if partial else ''}"

    def _decode_language(self, encoded_language: str):
        language, reviewed = encoded_language.split("_")
        return language, reviewed == "reviewed"

    def load_data(
        self,
        language: str,
        reviewed: bool,
        with_english=False,
        cleaned=False,
        partial=False,
        merged=False,
    ):
        # assert language in self.languages, f"Language {language} not supported"
        fn = self._encode_language(language, reviewed, partial)
        eng, lang = self._preprocess_json(language, pj(self.data_folder, f"{fn}.json"))

        if cleaned:
            eng = [self.clean_entity(e) for e in eng]
            lang = [self.clean_entity(e) for e in lang]

        if merged:
            eng = [self.merge_entity(e) for e in eng]
            lang = [self.merge_entity(e) for e in lang]

        if with_english:
            return eng, lang

        return lang


    def load_english_data(self, reviewed: bool=False, cleaned=False, merged=False):
        english_results = {}
        for language in LANGUAGES:
            eng, _ = self.load_data(language, reviewed, cleaned=cleaned, merged=merged, with_english=True)
            english_results[language] = eng
        return english_results


    @staticmethod
    def merge_entity(e):
        # remove the entities that are not in the SLOTS_MERGED
        result = []
        for annotator_result in e["annotations"]:
            annotator_merged_result = []
            for r in annotator_result["result"]:
                # if label is in the SLOTS_MERGED_MAPPER
                if r["value"]["labels"][0] in SLOTS_MERGED_MAPPER:
                    annotator_merged_result.append(r)
                elif r["value"]["labels"][0] in MERGE_MAPPER:
                    annotator_merged_result.append({
                        "id": r["id"],
                        "value": {
                            "start": r["value"]["start"],
                            "end": r["value"]["end"],
                            "text": r["value"]["text"],
                            "labels": [MERGE_MAPPER[r["value"]["labels"][0]]]
                        }
                    })

            annotator_result["result"] = annotator_merged_result
            result.append(annotator_result)
       
        e["annotations"] = result
        return e


    @staticmethod
    def clean_entity(e):
        return {
            "id": e["id"],
            "result": [
                [anno["id"]] + [r["value"] for r in anno["result"]]
                for anno in e["annotations"]
            ],
            "agreement": e["agreement"],
            "text": e["data"]["text"],
            "inner_id": e["inner_id"],
            "total_annotations": e["total_annotations"],
        }


from enum import Enum

import numpy as np
import pandas as pd


class IntentDataManager:
    # data/UtteranceGen/submission/amh/intent_data_Amharic.xlsx
    def __init__(self, data_folder="data/UtteranceGen/submission") -> None:
        self.data_folder = data_folder

    def load_data(self, language: str):
        assert language in LANGUAGES, f"Language {language} not supported"

        # print("####", language)
        df = pd.read_excel(
            pj(
                self.data_folder,
                f"{language}/intent_data_{LANGUAGES_MAPPER[language]}.xlsx",
            )
        )
        # print(df.columns)
        df.rename(
            columns={
                "utterance generation in an African language": "text",
                "text": "english",
            },
            inplace=True,
        )
        df.drop(columns=["QA Status", "Annotator", "english"], inplace=True)
        # print(df.columns)
        df["text"] = df["text"].astype(str).replace("\n", "")
        
        if language == "twi":
            df["intent"] = df["intent"].str.replace("book_ahɔhodan", "book_hotel")

        return df

    def load_english_data(self):
        langs = []
        for language in LANGUAGES:
            print(language)

            # Create an ExcelFile object
            excel_file = pd.ExcelFile(pj(
                        self.data_folder,
                        f"{language}.xlsx",
                    ))
            sheet_names = excel_file.sheet_names
            print(sheet_names, end="\t")

            for sheet in range(1, 4):
                table = pd.read_excel(
                    pj(
                        self.data_folder,
                        f"{language}.xlsx",
                    ),
                    sheet_name=sheet_names[sheet],
                )
                # print(sheet_names[sheet], table.columns.tolist())
            
                # drop any col has Unnamed in it
                # AttributeError: Can only use .str accessor with string values!
                # if language == "kin":
                #     table["split"] = "train"
                #     # move the split column to the first column
                #     cols = table.columns.tolist()
                #     cols = cols[-1:] + cols[:-1]
                #     table = table[cols]
                if len(table.columns) > 6:
                    unnamed_cols = [col for col in table.columns if "Unnamed" in col]
                    if "Comments" in table.columns:
                        unnamed_cols.append("Comments")
                    table.drop(unnamed_cols, axis=1, inplace=True)
                table.columns = ["split", "domain", "intent", "text", "generated", "English translation"]
                table["language"] = language
                langs.append(table)

                # sample = table.sample(1).to_dict(orient="records")[0]
                # print("English", sample["English translation"])

        df = pd.concat(langs)
        return df

import re

class SlotDataManager:
    def __init__(self, data_folder="./data/output") -> None:
        self.data = {}
        self.languages = LANGUAGES + ["eng", "clinc", "clinc+extend", "eng+40shots", "eng+23shots"] + [lang+"_eng" for lang in LANGUAGES]
        # self.languages_mapper = LANGUAGES_MAPPER | {"eng": "English"}
        self.data_folder = data_folder
        # for language in self.languages:
        #     self.data[language] = pd.read_csv(pj(data_folder, f"{language}.csv"))

    @staticmethod
    def parse_logical_form(logical_form: str) -> str:
        """Convert logical form to target format."""
        # [IN:balance [SL:ACCOUNT_TYPE àpò ìfowópamọ́ Dorm] ] => ACCOUNT_TYPE: àpò ìfowópamọ́ Dorm
        # [IN:translate [SL:DISH_OR_FOOD nemi] [SL:LANGUAGE_NAME fɔ̃gbe] ] => DISH_OR_FOOD: nemi $$ LANGUAGE_NAME: fɔ̃gbe
        parts = []
        for part in re.findall(r"\[SL:([A-Z_]+) ([^\]]+)\]", logical_form):
            label, value = part
            parts.append(f"{label}: {value}")
        return " $$ ".join(parts)

    def get_fewshot_examples(self, language, shot_count = 40):
        # Few Shot
        import random
        random.seed(2025)
        shuffled_intents = [i for i in INTENTS]
        random.shuffle(shuffled_intents)
        example_data = []
        print(shuffled_intents)
        train_df = self.load_data(language, split="train")
        # seqc: shot_count = 40 intent
        # seqc: shot_count = 10 domain (5 shots)
        # slot: shot_count = 23 slot type
        # shot_count = 5 random
        train_df.fillna("", inplace=True)
        train_df = train_df[train_df["spans"] != ""]
        # spans must contain $$
        # print(train_df["spans"])
        # train_df = train_df[train_df["spans"].apply(lambda x: "," in x)]
        # domain,intent,raw,text,language,spans,logical_form
        if shot_count == 40:
            for intent in INTENTS:
                line = train_df[train_df["intent"] == intent]
                line = line[line["text"] != ""]
                line = line.iloc[0]
                example_data.append(line) # {"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
        elif shot_count == 10:
            print(len(train_df["domain"].unique()), train_df["domain"].unique())
            for domain in train_df["domain"].unique():
                line = train_df[train_df["domain"] == domain]
                line = line[line["text"] != ""]
                line = line.iloc[0]
                example_data.append(line) #{"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
        elif shot_count == 23:
            # select first entity covers 23 slot types
            existed_slots = set()
            for _, line in train_df.iterrows():
                # 17:27:SL:MONEY,36:51:SL:BANK_NAME
                if isinstance(line["spans"], float):
                    continue
                slots = [slot.split(":")[-1] for slot in line["spans"].split(",")]
                for slot in slots:
                    if slot not in existed_slots:
                        example_data.append(line) # {"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
                        existed_slots |= set(slots)
                        break
                if len(existed_slots) >= 23:
                    break
        else:
            for intent in shuffled_intents[:shot_count]:
                line = train_df[train_df["intent"] == intent]
                line = line[line["xtreme-up"] != ""]
                line = line.iloc[0]
                example_data.append(line) # {"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})

        example_data = pd.concat(example_data, axis=1).T
        return example_data

    def load_data(self, language: str, split: str = "full", seed: int = 42):
        # assert language in self.languages, f"Language {language} not supported"
        assert split in ["full", "train", "dev", "test", "split"], f"Split {split} not supported"

        # lazy loading
        if self.data.get(language, None) is None:
            self.data[language] = pd.read_csv(pj(self.data_folder, f"{language}.csv"))
            self.data[language]["xtreme-up"] = self.data[language]["logical_form"].apply(self.parse_logical_form)
            self.data[language]["source_language"] = language

        data = self.data[language]

        if split == "full":
            return data

        if "split" in data.columns:
            splited_data = {}
            for s in ["train", "dev", "test"]:
                splited_data[s] = data[data["split"] == s].copy()
                # splited_data[s].drop(columns=["split"], inplace=True)
        else:
            if language == "eng":
                splited_data = self._split_english_data(data, seed)
            else:
                splited_data = self._split_data(data, seed)

        if split == "split":
            return splited_data

        return splited_data[split]


    def load_all_data(self, split: str = "full", seed: int = 42):
        results = {"train": [], "dev": [], "test": []}
        for lan in sorted(LANGUAGES + ["eng"]):
            merged = self.load_data(lan, split="split", seed=seed)
            for s in ["train", "test", "dev"]:
                results[s].append(merged[s])
        for s in ["train", "test", "dev"]:
            results[s] = pd.concat(results[s], ignore_index=True)
            print(s, results[s].shape)

        assert split != "full"

        if split == "split":
            return results

        return results[split]


    def _split_english_data(self, data: pd.DataFrame, seed: int):
        np.random.seed(seed)

        test_data = pd.DataFrame()
        for intent_type in INTENTS:
            intent_data = data[data["intent"] == intent_type]

            # every language has at least one sample
            test_samples = intent_data.groupby("language", group_keys=False).apply(lambda x: x.sample(n=1, replace=False))
            
            if intent_data["language"].nunique() < 16:
                print(f"Missing {intent_type} with", set(LANGUAGES) - set(intent_data["language"].tolist()))
                test_data = pd.concat([test_data, test_samples])
            else:
                test_data = pd.concat([test_data, test_samples])

        # Remove test samples from original data to get training candidates
        train_candidates = data[~data.index.isin(test_data.index)]

        # Split remaining data into train and dev
        dev_data = pd.DataFrame()
        for intent_type in INTENTS:
            intent_data = train_candidates[train_candidates["intent"] == intent_type]
            if len(intent_data) > 0:
                # Take 10% for dev set
                dev_samples = intent_data.sample(n=max(1, int(0.1 * len(intent_data))), random_state=seed)
                dev_data = pd.concat([dev_data, dev_samples])

        # Final train set excludes both test and dev samples
        train_data = train_candidates[~train_candidates.index.isin(dev_data.index)]

        return {
            "train": train_data, 
            "dev": dev_data,
            "test": test_data,
        }

    def _split_data(self, data: pd.DataFrame, seed: int):
        np.random.seed(seed)

        language = data["source_language"].iloc[0]

        train_data = pd.DataFrame()
        dev_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for intent_type in INTENTS:
            intent_data = data[data["intent"] == intent_type]

            # Shuffle the data
            intent_data = intent_data.sample(frac=1, random_state=seed).reset_index(
                drop=True
            )

            # Split the data
            # train_samples = intent_data.iloc[:56] if language != "clinc" else intent_data.iloc[:26]
            # dev_samples = intent_data.iloc[56:64] if language != "clinc" else intent_data.iloc[:60]

            if "clinc" in language:
                if "extend" in language:
                    train_samples = intent_data.iloc[27:54]
                    dev_samples = intent_data.iloc[60:64]
                else:
                    train_samples = intent_data.iloc[:27]
                    dev_samples = intent_data.iloc[56:60]
            else:
                train_samples = intent_data.iloc[:56]
                dev_samples = intent_data.iloc[56:64]

            test_samples = intent_data.iloc[64:]

            train_data = pd.concat([train_data, train_samples])
            dev_data = pd.concat([dev_data, dev_samples])
            test_data = pd.concat([test_data, test_samples])

        return {"train": train_data, "dev": dev_data, "test": test_data}

    TOKENC_PROMPT = None
    """Identify all named entities in the providing sentence according to the available entity types. Use `$$` as a separator between each identified named entity and corresponding content from the sentence. Only return the listed named entities without providing any additional commentary.

    # Named Entities Types to Identify
    {entity_types}

    # Output Format
    - List all the named entities found in the passage provided by the user. 
    - Separate the named entities using a `$$` symbol.
    - Only return the entity list, without any prefix or explanation. 

    # Format Example:
    Sentence: John went to Paris and paid 100 dollars at an Awater restaurant.
    Output: PERSONAL_NAME: John $$ CITY_OR_PROVINCE: Paris $$ MONEY: 100 $$ RESTAURANT_NAME: Awater

    Please note that the entities should match the listed types and unstated entities should not be included in the response. If no entities are found, return `$$` only.

    Sentence: {text}"""

    SEQC_PROMPT = None
    """Classify the sentence by intent, selecting from the available categories. Only return the chosen intent category without additional commentary or formatting.

    # Intent Categories
    {intent_types}

    # Output Format
    - Return the only one matching intent category from the list above. 
    - No additional text or punctuation should be included in the output. 

    # Format Example:
    Sentence: Can you tell me the weather forecast for today?
    Output: weather

    Sentence: {text}"""


    def generate_dataset_info(self, output_dir: str = "data/sft-dataset"):
        """Generate dataset_info.json for SFT training"""
        dataset_info = {}
        
        # Generate configurations for each language and task
        for lang in LANGUAGES + ["eng"]:
            # Token classification task
            dataset_name = f"sft_{lang}_tokenc"
            dataset_info[dataset_name] = {
                "file_name": f"{lang}_tokenc.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system"
                }
            }
            
            # Sequence classification task
            dataset_name = f"sft_{lang}_seqc"
            dataset_info[dataset_name] = {
                "file_name": f"{lang}_seqc.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system"
                }
            }

        # Save dataset info
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
        return dataset_info

    def get_tokenc_prompt(self, text: str, round: int = 1) -> str:
        """Generate token classification prompt"""
        return self.TOKENC_PROMPT.format(
            entity_types=", ".join(SLOTS_MERGED),
            text=text, round=round
        )

    def get_seqc_prompt(self, text: str, round: int = 1) -> str:
        """Generate sequence classification prompt"""
        return self.SEQC_PROMPT.format(
            intent_types=", ".join(INTENTS),
            text=text, round=round
        )

    def generate_sft_data(self, language: str, split: str = "full") -> dict:
        """Generate SFT data for both tasks"""
        data = self.load_data(language, split=split)
        
        tokenc_samples = []
        seqc_samples = []
        
        for _, row in data.iterrows():
            # Token classification samples
            tokenc_samples.append({
                "instruction": self.get_tokenc_prompt(row['text']),
                "input": "",
                "output": row['xtreme-up'],
                "system": f"You are an expert in {LANGUAGES_MAPPER[language]} language processing, specialized in named entity recognition."
            })
            
            # Sequence classification samples
            seqc_samples.append({
                "instruction": self.get_seqc_prompt(row['text']),
                "input": "",
                "output": row['intent'],
                "system": f"You are an expert in {LANGUAGES_MAPPER[language]} language processing, specialized in intent classification."
            })

        return {
            "tokenc": tokenc_samples,
            "seqc": seqc_samples
        }