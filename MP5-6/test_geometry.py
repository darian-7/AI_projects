from geometry import *
import argparse
import configparser
import pickle


def create_alien(map_name, configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    lims = eval(config.get(map_name, 'Window'))
    window = eval(config.get(map_name, 'Window'))
    centroid = eval(config.get(map_name, 'StartPoint'))
    widths = eval(config.get(map_name, 'Widths'))
    alien_shape = 'Ball'
    lengths = eval(config.get(map_name, 'Lengths'))
    alien_shapes = ['Horizontal', 'Ball', 'Vertical']
    obstacles = eval(config.get(map_name, 'Obstacles'))
    boundary = [(0, 0, 0, lims[1]), (0, 0, lims[0], 0), (lims[0], 0, lims[0], lims[1]), (0, lims[1], lims[0], lims[1])]
    obstacles.extend(boundary)
    alien = Alien(centroid, lengths, widths, alien_shapes, alien_shape, window)
    return alien


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS440 MP5/6 Robotics')
    parser.add_argument('--map_config_path', type=str, default='maps/test_config.txt')
    parser.add_argument('--test_data_path', type=str, default="new_geometry_test_data.pkl",
                        help='path to the pickle file of new geometry test data ')
    args = parser.parse_args()
    # from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
    #     alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints

    def test_alien_touch_wall(map_name, alien_config, walls, truth):
        alien = create_alien(map_name, args.map_config_path)
        alien.set_alien_config(alien_config)
        result = does_alien_touch_wall(alien, walls)

        assert result == truth, \
            f'In map {map_name}, does_alien_touch_wall(alien, walls) with alien config {alien_config} ' \
            f'returns {result}, expected: {truth}'

    def test_is_point_in_polygon(point, polygon, truth):
        result = is_point_in_polygon(point, polygon)
        assert result == truth, \
            f'does_alien_touch_wall(point, polygon) with point {point} and polygon {polygon} ' \
            f'returns {result}, expected: {truth}'

    def test_alien_path_touch_wall(map_name, alien_config, walls, waypoint, truth):
        alien = create_alien(map_name, args.map_config_path)
        alien.set_alien_config(alien_config)
        result = does_alien_path_touch_wall(alien, walls, waypoint)

        assert result == truth, \
            f'In map {map_name}, does_alien_path_touch_wall(alien, walls, waypoint) ' \
            f'with alien config {alien_config} and waypoint {waypoint} ' \
            f'returns {result}, expected: {truth}'


    with open(args.test_data_path, 'rb') as f:
        all_global_tests = pickle.load(f)

    for map_name, test_data in all_global_tests.items():
        walls = test_data['walls']
        for alien_config, truth in test_data['does_alien_touch_wall']:
            test_alien_touch_wall(map_name, alien_config, walls, truth)
        for point, polygon, truth in test_data['is_point_in_polygon']:
            test_is_point_in_polygon(point, polygon, truth)
        for alien_config, waypoint, truth in test_data['does_alien_path_touch_wall']:
            test_alien_path_touch_wall(map_name, alien_config, walls, waypoint, truth)
        print(f'Map {map_name} Geometry Test Passed!')

    print("Geometry tests passed\n")
