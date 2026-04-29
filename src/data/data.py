import json
import numpy as np
from Bio import SeqIO

import base.data
from src.config import (
    DISPROT_DATABASE_FILE,
    PROCESSED_DATA_DIR,
    ALL_DATA_TEMPLATE,
    DATASETS,
    CAID_BENCHMARK,
    CUT_DATA_TEMPLATE,
)
from Bio import SeqIO
from base.utils.io import format_filename

SEQUENCES = "sequences"
DISORDER_REGIONS = "disorder region"
ID = "id"


class IDPData(base.data.Data):
    def extend(self, X, y):
        pass

    def subset(self, indices) -> tuple:
        pass


def get_data() -> IDPData:
    data_dict = get_disprot_cut_data()
    data = IDPData(data_dict[SEQUENCES], data_dict[DISORDER_REGIONS])
    return data


def get_disprot_cut_data():
    with open(
        format_filename(PROCESSED_DATA_DIR, CUT_DATA_TEMPLATE, dataset=DATASETS[1]),
        "r",
    ) as read_file:
        data = json.load(read_file)
        return data


def get_disprot_data():
    with open(
        format_filename(PROCESSED_DATA_DIR, ALL_DATA_TEMPLATE, dataset=DATASETS[1]),
        "r",
    ) as read_file:
        data = json.load(read_file)
        return data


def process_disprot_data(cut_length=400, alpha=1 / 2, save=True):
    all_data = process_disprot_all_data(save=save)
    cut_data = process_disprot_cut_data(
        all_data=all_data, cut_length=cut_length, alpha=alpha, save=save
    )
    return cut_data


def process_disprot_all_data(save=True):
    data = {}  # main dictionary for saving sequences and their labels
    with open(DISPROT_DATABASE_FILE, "r") as disprot_database:
        total_proteins_data = json.load(disprot_database)

        id_list = []
        sequence_list = []
        disorder_regions_list = []
        for protein_data in total_proteins_data["data"]:
            sequence = protein_data["sequence"]
            disprot_id = protein_data["disprot_id"]

            disorder_region = np.zeros(protein_data["length"])
            for region in protein_data["regions"]:
                if (
                    region["term_name"] == "disorder"
                    or region["term_name"] == "disorder to order"
                ):
                    for i in range(region["start"] - 1, region["end"]):
                        disorder_region[i] = 1
            disorder_region = disorder_region.astype(int).tolist()

            sequence_list.append(sequence)
            disorder_regions_list.append(disorder_region)
            id_list.append(disprot_id)

        data[SEQUENCES] = sequence_list
        data[DISORDER_REGIONS] = disorder_regions_list
        data[ID] = id_list

        if save:
            with open(
                format_filename(
                    PROCESSED_DATA_DIR, ALL_DATA_TEMPLATE, dataset=DATASETS[1]
                ),
                "w",
            ) as write_file:
                json.dump(data, write_file)
    return data


def process_disprot_cut_data(all_data, cut_length=400, alpha=1 / 2, save=True):
    cut_data = {}

    sequence_list = all_data[SEQUENCES]
    disorder_regions_list = all_data[DISORDER_REGIONS]
    sequence_ids = all_data[ID]
    n = len(all_data[SEQUENCES])

    temp_sequence_list = []
    temp_disorder_regions_list = []

    for i in range(n):
        if sequence_ids[i] in get_caid_ids():
            continue

        if len(sequence_list[i]) >= cut_length:
            temp_sequence = sequence_list[i]
            temp_disorder_region = disorder_regions_list[i].copy()
            while len(temp_sequence) >= cut_length:
                before_length = len(temp_sequence)
                temp_sequence_list.append(temp_sequence[:cut_length])
                temp_disorder_regions_list.append(temp_disorder_region[:cut_length])
                temp_sequence = temp_sequence[int(before_length * (1 - alpha)) :]
                temp_disorder_region = temp_disorder_region[
                    int(before_length * (1 - alpha)) :
                ]
        else:
            temp_sequence_list.append(sequence_list[i])
            temp_disorder_regions_list.append(disorder_regions_list[i].copy())

    cut_data[SEQUENCES] = temp_sequence_list
    cut_data[DISORDER_REGIONS] = temp_disorder_regions_list

    if save:
        with open(
            format_filename(PROCESSED_DATA_DIR, CUT_DATA_TEMPLATE, dataset=DATASETS[1]),
            "w",
        ) as write_file:
            json.dump(cut_data, write_file)

    return cut_data


def get_caid_ids():
    caid_ids = []
    with open(CAID_BENCHMARK) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            caid_ids.append(record.id)
    return caid_ids


def get_caid_data():
    data = {}
    with open(CAID_BENCHMARK) as handle:
        sequence_list = []
        sequence_ids = []
        disorder_regions_list = []
        for record in SeqIO.parse(handle, "fasta"):
            str_seq = str(record.seq)
            for i in range(len(str_seq)):
                if str_seq[i] == "1" or str_seq[i] == "0":
                    break

            sequence_list.append(str_seq[:i])
            sequence_ids.append(record.id)

            str_region = str_seq[i:]
            list_region = []
            for c in str_region:
                list_region.append(int(c))
            disorder_regions_list.append(list_region)

        data[SEQUENCES] = sequence_list
        data[DISORDER_REGIONS] = disorder_regions_list
        data[ID] = sequence_ids

    return data
