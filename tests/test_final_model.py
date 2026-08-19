import numpy as np
import pytest

from final_model import cout_metier, seuil_optimal


class TestCoutMetier:
    def test_aucune_erreur(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert cout_metier(y_true, y_pred) == 0

    def test_un_faux_negatif(self):
        # client à risque prédit comme sain -> coûte cout_fn (10 par défaut)
        y_true = np.array([1])
        y_pred = np.array([0])
        assert cout_metier(y_true, y_pred) == 10

    def test_un_faux_positif(self):
        # bon client prédit comme à risque-> coûte cout_fp (1 par défaut)
        y_true = np.array([0])
        y_pred = np.array([1])
        assert cout_metier(y_true, y_pred) == 1

    def test_asymetrie_fn_plus_couteux_que_fp(self):
        y_true = np.array([1, 0])
        y_pred_fn = np.array([0, 0])  # 1 FN, 0 FP
        y_pred_fp = np.array([1, 1])  # 0 FN, 1 FP
        assert cout_metier(y_true, y_pred_fn) > cout_metier(y_true, y_pred_fp)

    def test_couts_personnalises(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0, 1])  # 1 FN + 1 FP
        assert cout_metier(y_true, y_pred, cout_fn=5, cout_fp=2) == 7

    def test_melange_vrais_et_faux(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])  # 1 FN (idx 1), 1 FP (idx 3)
        assert cout_metier(y_true, y_pred) == 10 * 1 + 1 * 1


class TestSeuilOptimal:
    def test_separation_parfaite_donne_seuil_qui_minimise_le_cout(self):
        # classes séparées par la proba : le seuil optimal doit conduire à un coût métier nul
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        seuil = seuil_optimal(y_true, y_pred_proba)
        y_pred = (y_pred_proba > seuil).astype(int)

        assert cout_metier(y_true, y_pred) == 0

    def test_seuil_dans_lintervalle_des_probas(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.2, 0.4, 0.6, 0.8])

        seuil = seuil_optimal(y_true, y_pred_proba)

        assert 0.0 <= seuil <= 1.0

    def test_favorise_le_recall_vu_le_cout_asymetrique(self):
        # avec FN 10x plus coûteux qu'un FP, le seuil optimal doit rester bas quitte a générer des FP plutôt que de râter des défauts
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred_proba = np.array([0.15, 0.25, 0.35, 0.55, 0.5, 0.6])

        seuil = seuil_optimal(y_true, y_pred_proba)
        y_pred = (y_pred_proba > seuil).astype(int)

        recall = y_pred[y_true == 1].sum() / (y_true == 1).sum()
        assert recall == 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
