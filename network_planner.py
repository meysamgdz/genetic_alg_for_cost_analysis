import numpy as np
import matplotlib.pyplot as plt

class NetworkPlanner:
    """A class to plan and optimize wireless network access point placement."""

    def __init__(self, xy, frequency, pl_exponent, num_ap=1, tx_power=23, resolution_factor=1):
        """
        Initialize the network planner.

        Args:
            xy (tuple): Dimensions of the area (width, height) in meters
            frequency (float): Operating frequency in Hz
            pl_exponent (float): Path loss exponent
            num_ap (int): Number of access points
            tx_power (float): Transmit power in dBm
            resolution_factor (int): Resolution factor for the simulation grid
        """
        self.x, self.y = xy
        self.resolution_factor = resolution_factor
        self.num_ap = num_ap
        self.tx_power = tx_power
        self.frequency = frequency
        self.pl_exponent = pl_exponent

    def calculate_pathloss(self, distance):
        """
        Calculate path loss for a given distance.

        Args:
            distance (float or ndarray): Distance(s) from transmitter in meters

        Returns:
            float or ndarray: Path loss in dB
        """
        pl0 = -147.55 + 20 * np.log10(self.frequency)  # Free space path loss at 1m
        pl = pl0 + 10 * self.pl_exponent * np.log10(distance)
        return pl

    def map_pathloss(self, ap_locs=(), nth=0):
        """
        Create a path loss map for given AP locations.

        Args:
            ap_locs (tuple): Tuple of (x_coords, y_coords) for AP locations
            nth (int): Which AP's path loss to return (0=strongest)

        Returns:
            tuple: (AP locations, path loss map)
        """
        if ap_locs:
            x_ap, y_ap = ap_locs
            if len(x_ap.shape) != 3:
                x_ap = np.tile(x_ap[:, None, None],
                               (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
                y_ap = np.tile(y_ap[:, None, None],
                               (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
        else:
            x_ap = np.random.uniform(0, self.x, self.num_ap)
            ind_temp = np.argsort(x_ap)
            x_ap = np.sort(x_ap)
            y_ap = np.random.uniform(0, self.y, self.num_ap)
            y_ap = y_ap[ind_temp]
            x_ap = np.tile(x_ap[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
            y_ap = np.tile(y_ap[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))

        x_ue, y_ue = np.meshgrid(np.linspace(0, self.x, self.resolution_factor * self.x),
                                 np.linspace(0, self.y, self.resolution_factor * self.y))
        distance_vec = ((x_ue - x_ap) ** 2 + (y_ue - y_ap) ** 2) ** 0.5
        pl_vec = self.calculate_pathloss(distance_vec)
        pl_sorted = np.sort(pl_vec, axis=0)
        pl_map = pl_sorted[nth, :, :]
        return (x_ap, y_ap), pl_map

    def generate_heatmap(self, ap_locs=(), nth=0):
        """Generate a heatmap visualization of path loss."""
        if ap_locs:
            x_ap, y_ap = ap_locs
            if len(x_ap.shape) != 3:
                x_ap = np.tile(x_ap[:, None, None],
                               (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
                y_ap = np.tile(y_ap[:, None, None],
                               (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
            (x_ap, y_ap), pl_map = self.map_pathloss((x_ap, y_ap), nth=nth)

        x_ue, y_ue = np.meshgrid(np.linspace(0, self.x, self.resolution_factor * self.x),
                                 np.linspace(0, self.y, self.resolution_factor * self.y))
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x_ue, y_ue, -pl_map, cmap='RdBu', vmin=-75, vmax=-55)
        ax.scatter(x_ap[:, 0, 0], y_ap[:, 0, 0])
        ax.axis([x_ue.min(), x_ue.max(), y_ue.min(), y_ue.max()])
        ax.set(title=f'Path loss of AP {nth}')
        fig.colorbar(c, ax=ax)
        return fig, ax

    def create_mask(self):
        """Create a mask representing important areas in the facility."""
        ind_x, ind_y = self.resolution_factor * self.x, self.resolution_factor * self.y
        edge_coeff = 0.15
        mask = np.zeros((ind_x, ind_y))
        mask[int(edge_coeff * ind_x):int((1 - edge_coeff) * ind_x), int(edge_coeff * ind_y)] = 1
        mask[int(edge_coeff * ind_x):int((1 - edge_coeff) * ind_x), int((1 - edge_coeff) * ind_y)] = 1
        mask[int(edge_coeff * ind_x), int(edge_coeff * ind_y):int((1 - edge_coeff) * ind_y)] = 1
        mask[int(0.5 * ind_x), int(edge_coeff * ind_y):int((1 - edge_coeff) * ind_y)] = 1
        mask[int((1 - edge_coeff) * ind_x), int(edge_coeff * ind_y):int((1 - edge_coeff) * ind_y)] = 1
        return mask

    def uniform_ap_distribution(self):
        """Distribute APs uniformly along important paths."""
        x_ap, y_ap = [], []
        edge_coeff = 0.1
        ap_density = (2 * (1 - 2 * (1 - edge_coeff)) * self.x + 3 * (1 - 2 * (1 - edge_coeff)) * self.y) / (
                    self.num_ap + 1)

        # Add APs along horizontal paths
        x_ap += list(np.arange(edge_coeff * self.x + ap_density, (1 - edge_coeff) * self.x, ap_density))
        y_ap += [edge_coeff * self.y] * len(x_ap)
        x_ap += list(np.arange(edge_coeff * self.x + ap_density, (1 - edge_coeff) * self.x, ap_density))
        y_ap += [(1 - edge_coeff) * self.y] * (len(x_ap) - len(y_ap))

        # Add APs along vertical paths
        y_ap += list(np.arange(edge_coeff * self.y + ap_density, (1 - edge_coeff) * self.y, ap_density))
        x_ap += [edge_coeff * self.x] * (len(y_ap) - len(x_ap))
        y_ap += list(np.arange(edge_coeff * self.y + ap_density, (1 - edge_coeff) * self.y, ap_density))
        x_ap += [0.5 * self.x] * (len(y_ap) - len(x_ap))
        y_ap += list(np.arange(edge_coeff * self.y + ap_density, (1 - edge_coeff) * self.y, ap_density))
        x_ap += [(1 - edge_coeff) * self.x] * (len(y_ap) - len(x_ap))

        # Fill remaining APs randomly if needed
        if len(x_ap) != self.num_ap:
            x_ap += list(np.random.uniform(0, self.x, self.num_ap - len(x_ap)))
            y_ap += list(np.random.uniform(0, self.y, self.num_ap - len(y_ap)))

        x_ap, y_ap = np.array(x_ap), np.array(y_ap)
        x_ap = np.tile(x_ap[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
        y_ap = np.tile(y_ap[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
        return (x_ap, y_ap)

    def lattice_distribution(self):
        """Distribute APs in a regular lattice pattern."""
        N_new = int(np.ceil(np.sqrt(self.num_ap))) ** 2
        n_cols = int(np.sqrt(N_new))
        n_rows = N_new // n_cols

        step_x = self.x / n_cols
        step_y = self.y / n_rows

        x_coords = np.linspace(0, self.x, n_cols, endpoint=False)
        y_coords = np.linspace(0, self.y, n_rows, endpoint=False)

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        x_coords = x_grid.flatten()[:self.num_ap]
        y_coords = y_grid.flatten()[:self.num_ap]

        x_ap = np.tile(x_coords[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
        y_ap = np.tile(y_coords[:, None, None], (1, self.resolution_factor * self.y, self.resolution_factor * self.x))
        return (x_ap, y_ap)

    def genetic_optimization(self, threshold=45, init_iter=100, dropout=0.8, nth=0):
        """
        Optimize AP placement using a genetic algorithm.

        Args:
            threshold (float): Path loss threshold for coverage
            init_iter (int): Number of initial iterations
            dropout (float): Fraction of solutions to drop each iteration
            nth (int): Which AP's path loss to consider

        Returns:
            tuple: (scores, ap_locations, coverage_pathloss, nth_pathloss)
        """
        ap_locs, pl_agg, cov_thresh = [], [], 5
        pl_cov = np.empty((0, self.resolution_factor * self.y, self.resolution_factor * self.x))
        pl_nth = np.empty((0, self.resolution_factor * self.y, self.resolution_factor * self.x))
        for i in range(init_iter):
            ap_locs_temp, pl_temp = self.map_pathloss()
            pl_cov = np.append(pl_cov, pl_temp[None, :, :], axis=0)
            _, pl_temp = self.map_pathloss(ap_locs_temp, nth=nth)
            pl_nth = np.append(pl_nth, pl_temp[None, :, :], axis=0)
            ap_locs += [ap_locs_temp]
        # add the two type of uniform designs 1) uniformly over the AGV path 2) uniformly in the entire factory
        ap_locs_temp = self.uniform_ap_distribution()
        ap_locs += [ap_locs_temp]
        _, pl_temp = self.map_pathloss(ap_locs_temp)
        pl_cov = np.append(pl_cov, pl_temp[None, :, :], axis=0)
        _, pl_temp = self.map_pathloss(ap_locs_temp, nth=nth)
        pl_nth = np.append(pl_nth, pl_temp[None, :, :], axis=0)
        ap_locs_temp = self.lattice_distribution()
        ap_locs += [ap_locs_temp]
        _, pl_temp = self.map_pathloss(ap_locs_temp)
        pl_cov = np.append(pl_cov, pl_temp[None, :, :], axis=0)
        _, pl_temp = self.map_pathloss(ap_locs_temp, nth=nth)
        pl_nth = np.append(pl_nth, pl_temp[None, :, :], axis=0)

        mask = self.create_mask()  # this will help to implement the fitness function
        while True:
            max_score_1, max_score_2 = mask.sum(), self.resolution_factor ** 2 * self.x * self.y
            # max_score_1, max_score_2 = 1, 1000
            scores = list(np.array(
                [np.sum(pl_nth[i, :, :].T * mask < threshold * mask) for i in range(pl_nth.shape[0])]) / max_score_1 \
                          * np.array(
                [np.sum(pl_cov[i, :, :].T < threshold - cov_thresh) for i in range(pl_cov.shape[0])]) / max_score_2)
            # sort the scores
            indexes = np.argsort(scores)[::-1]
            sorted_scores = np.sort(scores)[::-1]
            # throwing away half of the solutions
            ap_locs = [ap_locs[i] for i in indexes[0:int((1 - dropout) * pl_cov.shape[0])]]
            pl_cov = pl_cov[indexes[0:int((1 - dropout) * pl_cov.shape[0])], :, :]
            pl_nth = pl_nth[indexes[0:int((1 - dropout) * pl_nth.shape[0])], :, :]
            keep = int((1 - dropout) * pl_cov.shape[0])
            print(f"Top {keep} scores to be kept for breeding and mutation:\n", sorted_scores[:keep])

            # survived designs breeding
            breed_rate = 3.5 / pl_cov.shape[0]
            rng = np.random.randint(0, pl_cov.shape[0], size=(int(breed_rate * pl_cov.shape[0] ** 2), 2))
            for i in rng:
                ap_locs_temp = tuple(0.5 * np.array(ap_locs[i[0]]) + 0.5 * np.array(ap_locs[i[1]]))
                ap_locs += [ap_locs_temp]
                _, pl_temp = self.map_pathloss(ap_locs_temp)
                pl_cov = np.append(pl_cov, pl_temp[None, :, :], axis=0)
                _, pl_temp = self.map_pathloss(ap_locs_temp, nth=nth)
                pl_nth = np.append(pl_nth, pl_temp[None, :, :], axis=0)

            # generating mutated version of some designs
            mutation_rate = 0.1
            rng = np.random.randint(0, pl_cov.shape[0], int(mutation_rate * pl_cov.shape[0]))
            for i in rng:
                ap_locs_temp = np.array(ap_locs[i])
                random_mutation = np.random.randint(0, 2, size=(ap_locs_temp.shape[0], ap_locs_temp.shape[1])) \
                                  * np.random.normal(0, 1, size=(ap_locs_temp.shape[0], ap_locs_temp.shape[1]))
                random_mutation = np.tile(random_mutation, (self.resolution_factor * self.y,
                                                            self.resolution_factor * self.x, 1, 1)).transpose(2, 3, 0,
                                                                                                              1)
                ap_locs_temp = tuple(np.array(ap_locs[i]) + random_mutation)
                ap_locs += [ap_locs_temp]
                _, pl_temp = self.map_pathloss(ap_locs_temp)
                pl_cov = np.append(pl_cov, pl_temp[None, :, :], axis=0)
                _, pl_temp = self.map_pathloss(ap_locs_temp, nth=nth)
                pl_nth = np.append(pl_nth, pl_temp[None, :, :], axis=0)

            print("Total number of designs: ", pl_cov.shape[0])
            if pl_cov.shape[0] < 5:
                scores = list(np.array(
                    [np.sum(pl_nth[i, :, :].T * mask < threshold * mask) for i in range(pl_nth.shape[0])]) / max_score_1 \
                              * np.array(
                    [np.sum(pl_cov[i, :, :].T < threshold - cov_thresh) for i in range(pl_cov.shape[0])]) / max_score_2)
                return scores, ap_locs, pl_cov, pl_nth
