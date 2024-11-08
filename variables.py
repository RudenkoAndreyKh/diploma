from pathlib import Path

# Мапа рослин
plantsMap = {
    0: 'African Violet (Saintpaulia ionantha)',
    1: 'Aloe Vera',
    2: 'Anthurium (Anthurium andraeanum)',
    3: 'Areca Palm (Dypsis lutescens)',
    4: 'Asparagus Fern (Asparagus setaceus)',
    5: 'Begonia (Begonia spp.)',
    6: 'Bird of Paradise (Strelitzia reginae)',
    7: 'Birds Nest Fern (Asplenium nidus)',
    8: 'Boston Fern (Nephrolepis exaltata)',
    9: 'Calathea',
    10: 'Cast Iron Plant (Aspidistra elatior)',
    11: 'Chinese evergreen (Aglaonema)',
    12: 'Chinese Money Plant (Pilea peperomioides)',
    13: 'Christmas Cactus (Schlumbergera bridgesii)',
    14: 'Chrysanthemum',
    15: 'Ctenanthe',
    16: 'Daffodils (Narcissus spp.)',
    17: 'Dracaena',
    18: 'Dumb Cane (Dieffenbachia spp.)',
    19: 'Elephant Ear (Alocasia spp.)',
    20: 'English Ivy (Hedera helix)',
    21: 'Hyacinth (Hyacinthus orientalis)',
    22: 'Iron Cross begonia (Begonia masoniana)',
    23: 'Jade plant (Crassula ovata)',
    24: 'Kalanchoe',
    25: 'Lilium (Hemerocallis)',
    26: 'Lily of the valley (Convallaria majalis)',
    27: 'Money Tree (Pachira aquatica)',
    28: 'Monstera Deliciosa (Monstera deliciosa)',
    29: 'Orchid',
    30: 'Parlor Palm (Chamaedorea elegans)',
    31: 'Peace lily',
    32: 'Poinsettia (Euphorbia pulcherrima)',
    33: 'Polka Dot Plant (Hypoestes phyllostachya)',
    34: 'Ponytail Palm (Beaucarnea recurvata)',
    35: 'Pothos (Ivy arum)',
    36: 'Prayer Plant (Maranta leuconeura)',
    37: 'Rattlesnake Plant (Calathea lancifolia)',
    38: 'Rubber Plant (Ficus elastica)',
    39: 'Sago Palm (Cycas revoluta)',
    40: 'Schefflera',
    41: 'Snake plant (Sanseviera)',
    42: 'Tradescantia',
    43: 'Tulip',
    44: 'Venus Flytrap',
    45: 'Yucca',
    46: 'ZZ Plant (Zamioculcas zamiifolia)'
}

diseaseOrHealthy = {
    0: 'Bacterial',
    1: 'Disease',
    2: 'Early_blight',
    3: 'Healthy',
    4: 'Late_blight',
    5: 'Leaf_curl',
    6: 'Mold',
    7: 'Septoria_leaf_spot',
    8: 'Spider_mites',
}

# Шлях до кореневої папки проєкту
root_dir = Path(__file__).resolve().parent

# Роздільна якість зображень для рослин
plant_resolution = 512

# Роздільна якість зображень для хвороб
disease_resolution = 224

# Шлях до файлу впізнавання рослин
plant_recognition_model_path = root_dir / "plant_recognition" / "plant_recognition_model.keras"

# Шлях до датасету рослин
plant_recognition_dataset_path = root_dir / "data" / "house_plant_species"


# Шлях до файлу впізнавання хвороб
disease_recognition_model_path = root_dir / "leaves_disease_recognition" / "disease_recognition_model.keras"

# Шлях до датасету хвороб
disease_recognition_dataset_path = root_dir / "data" / "leaves_for_disease_recognition"

batchSize = 256