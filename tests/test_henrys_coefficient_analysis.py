import pytest
from pytest import approx

import enose.helper.henrys_coefficient_analysis as hca


def test_create_weight_matrix():
    error = [0,1,2,4,5]
    assert all([a == b for a,b in zip(hca.create_weight_matrix(error, weight_type=None), [1, 1, 1, 1, 1])])
    assert all([a == b for a,b in zip(hca.create_weight_matrix(error, weight_type='error'), [1, 1, 0.5, 0.25, 0.2])])
    assert all([a == b for a,b in zip(hca.create_weight_matrix(error, weight_type='error_squared'), [1, 1, 0.25, 0.0625, 0.04])])


def test_calculate_r2_and_rmse():
    p = [1, 0]
    x_data = [0,1,2,3,4]
    y_data = [0,1,2,3,4]
    assert hca.calculate_r2_and_rmse(p, x_data, y_data) == (1, approx(0), 2)


def test_check_r2_and_rmse():
    r2, r2_min, rmse, rmse_min, y_bar = [0.98, 0.99, 0.1, 0.2, 3]
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type=None) == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='R2') == False
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='RMSE') == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='Either') == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='Both') == False

    r2, r2_min, rmse, rmse_min, y_bar = [0.99, 0.98, 3, 0.2, 3]
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type=None) == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='R2') == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='RMSE') == False
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='Either') == True
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='Both') == False

    r2, r2_min, rmse, rmse_min, y_bar = [0.99, 0.98, 0.1, 0.2, 3]
    assert hca.check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type='Both') == True


def test_calculate_kH():

    gas = 'test'
    ads_data = {'test_comp': [0, 1, 2, 3, 4, 5, 6],
        'test_mass': [1, 2, 3, 4, 5, 6, 7],
        'test_error': [0, 0, 0, 0, 0, 0, 0] }
    eval_type='R2'
    r2_min=0.90
    rmse_min=0.10
    weight_type='error'
    fixed_intercept=False

    p, max_comp, r2, rmse, i =  hca.calculate_kH(ads_data, gas, eval_type=eval_type, r2_min=r2_min, rmse_min=rmse_min, weight_type=weight_type, fixed_intercept=fixed_intercept)

    assert all(a == approx(b) for a,b in zip(p, [1,1]))
    assert max_comp == 6
    assert r2 == 1
    assert rmse == approx(0)
    assert i == 7


def test_calculate_kH_air():

    gas = 'test'
    ads_data = {'test_comp': [0, 1, 2, 3, 4, 5, 6],
        'O2_mass': [7, 6, 5, 4, 3, 2, 1],
        'N2_mass': [7, 6, 5, 4, 3, 2, 1],
        'O2_error': [0, 0, 0, 0, 0, 0, 0],
        'N2_error': [0, 0, 0, 0, 0, 0, 0] }
    weight_type='error'
    i = 7

    p, r2, rmse = hca.calculate_kH_air(ads_data, gas, i, weight_type=weight_type)

    assert all(a == approx(b) for a,b in zip(p, [-1,7]))
    assert r2 == 1
    assert rmse == approx(0)
