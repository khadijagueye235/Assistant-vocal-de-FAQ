"""Les réglages se remplacent par variables d'environnement."""

import importlib

import pytest

import src.config


@pytest.fixture
def recharger_config(monkeypatch):
    """Recharge src.config avec l'environnement modifié, puis restaure."""

    def _recharger(**variables):
        for nom, valeur in variables.items():
            monkeypatch.setenv(nom, valeur)
        return importlib.reload(src.config)

    yield _recharger
    monkeypatch.undo()
    importlib.reload(src.config)


def test_valeurs_par_defaut():
    assert src.config.LANGUE == "fr"
    assert src.config.ASR_PROMPT_INITIAL is None
    assert 0 < src.config.SEUIL_SIMILARITE < 1


def test_remplacement_par_environnement(recharger_config):
    config = recharger_config(
        WHISPER_MODELE="medium",
        ASR_BEAM_SIZE="1",
        SEUIL_SIMILARITE="0.7",
        ASR_PROMPT_INITIAL="Western Union, MTCN",
    )
    assert config.WHISPER_MODELE == "medium"
    assert config.ASR_BEAM_SIZE == 1  # converti en entier
    assert config.SEUIL_SIMILARITE == 0.7  # converti en décimal
    assert config.ASR_PROMPT_INITIAL == "Western Union, MTCN"


def test_chemins_existent():
    assert src.config.FAQ_CSV.is_file()
    assert src.config.AUDIO_TEST_DIR.is_dir()
