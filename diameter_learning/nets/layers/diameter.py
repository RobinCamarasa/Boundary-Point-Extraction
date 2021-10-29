"""Implement `diameter_learning.nets.layers.diameter` classes
"""
from itertools import product
import math
import torch


class CenterOfMass2DExtractor(torch.nn.Module):
    """Class that extend the `torch.nn.Module` and compute the center
    of mass of a segmentation slice wise
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extension of the `torch.nn.Module.forward` method

        :param x: Tensor of shape (nb, nf, nx, ny, nz)
            where nb is the batch size, nf the number of features,
            nx the x dimension, ny the y dimension and nz the z dimension
        :return: A complex tensor of shape (nb, nf, nz) corresponding
            to the coordinates of the slice wise center of mass of
            the segmentation where nb is the batch size, nf the number of
            features, and nz the z dimension
        """
        # Compute the total mass
        index_array = torch.arange(
                x.shape[2] * x.shape[3], dtype=torch.float32
                ).reshape(
                        x.shape[2], x.shape[3]
                        ).to(x.device)
        x_indices = torch.fmod(index_array, x.shape[3])
        y_indices = torch.div(
            index_array, x.shape[3], rounding_mode='floor'
            )
        mass = x.sum(list(range(2, len(x.shape) - 1)))

        center_of_mass = torch.complex(
                torch.zeros(mass.shape, dtype=torch.float32),
                torch.zeros(mass.shape, dtype=torch.float32)
                ).to(x.device)

        # Loop to get the center of mass
        for nb, nf, nz in product(
                range(x.shape[0]),
                range(x.shape[1]),
                range(x.shape[-1])
                ):
            center_of_mass[nb, nf, nz] = torch.complex(
                    (x[nb, nf, :, :, nz] * x_indices / mass[nb, nf, nz]).sum(),
                    (x[nb, nf, :, :, nz] * y_indices / mass[nb, nf, nz]).sum()
                    )
        return center_of_mass


class GaussianRadiusExtractor(torch.nn.Module):
    """Class that extend the `torch.nn.Module` and compute radiuses between
    the center of mass and the border of a segmentation with an angle
    step of `\frac{2\pi}{nb\_radiuses}`

    :param nb_radiuses: The number of radiuses
    :param sigma: The standard deviation of the gaussian used in the method
    :param epsilon: The parameter of the approximation
    """
    def __init__(
        self, nb_radiuses: int = 24, sigma: float = 0.1, epsilon=10**(-5)
    ):
        super().__init__()
        self.nb_radiuses = nb_radiuses
        self.sigma = sigma
        self.epsilon = 10**(-5)
        self.angles = [
                i * (2 * math.pi / self.nb_radiuses) - math.pi
                for i in range(self.nb_radiuses)
                ]
        self.gaussian = lambda x, angle: (
            1 / (
                self.sigma * math.sqrt(2 * math.pi)
            ) *
            torch.exp(- (x - angle) ** 2 / (2 * self.sigma ** 2))
            )

    def get_filter(
            self, center_of_mass: torch.Tensor,
            complex_plan: torch.Tensor,
            angle: float,
            ) -> torch.Tensor:
        """Get the filter defined in the method

        :param center_of_mass: A complex tensor of shape (nb, nf, nz)
            corresponding to the coordinates of the slice wise center of mass
            of the segmentation where nb is the batch size, nf the number of
            features, and nz the z dimension
        :param complex_plan: A complex tensor of shape (nx, ny) corresponding
            to complex plan coordinates where nx is the x dimension and ny
            the y dimension
        :param angle: An angle between 0 and `2\pi`
        :return: A tensor of shape (nb, nf, nx, ny, nz) that contains the
            filter for an angle alpha where nb is the batch size, nf the
            number of features, nx is the x dimension, ny the y dimension
            and nz the z dimension
        """
        # The different conditions represents the different case for the
        # angle due to the limit of the function torch.angle
        angle_filter = torch.zeros(
                center_of_mass.shape[:-1] + complex_plan.shape +
                (center_of_mass.shape[-1],)
            ).to(center_of_mass.device)
        for nb, nf, nz in product(
                range(center_of_mass.shape[0]),
                range(center_of_mass.shape[1]),
                range(center_of_mass.shape[-1])
                ):
            centered_plan = complex_plan - center_of_mass[nb, nf, nz]
            get_arg = lambda x: torch.atan2(x.imag, x.real)
            if angle > 0.5 * math.pi:
                angle_filter[nb, nf, :, :, nz] = self.gaussian(
                    get_arg(-centered_plan), angle - math.pi
                    )
            elif angle < - 0.5 * math.pi:
                angle_filter[nb, nf, :, :, nz] = self.gaussian(
                    get_arg(-centered_plan), angle + math.pi
                    )
            else:
                angle_filter[nb, nf, :, :, nz] = self.gaussian(
                    get_arg(centered_plan), angle
                    )
        return angle_filter

    def forward(
            self, x: torch.Tensor, center_of_mass: torch.Tensor
    ) -> torch.Tensor:
        """Extension of the `torch.nn.Module.forward` method

        :param x: Tensor of shape (nb, nf, nx, ny, nz)
            where nb is the batch size, nf the number of features,
            nx the x dimension, ny the y dimension and nz the z dimension
        :param center_of_mass: A complex tensor of shape (nb, nf, nz)
            corresponding to the coordinates of the slice wise center of mass
            of the segmentation where nb is the batch size, nf the number of
            features, and nz the z dimension
        :return: Tensor of shape (nb, nf, nr, nz) where nb is the batch
            size, nf the number of features, nr the number of radiuses and nz
            the z dimension
        """
        # Generate complex plan
        index_array = torch.arange(
            x.shape[2] * x.shape[3]
            ).reshape(
                x.shape[2], x.shape[3]
            )
        complex_plan = torch.complex(
            (index_array % x.shape[3]).float(),
            (index_array // x.shape[3]).float()
            ).to(x.device)

        # Output
        radiuses = torch.zeros(
            x.shape[:2] + (self.nb_radiuses, x.shape[-1])
            ).to(x.device)

        # Apply the different integrals to get the radiuses
        for i, angle in enumerate(self.angles):
            angle = i * (2 * math.pi / self.nb_radiuses) - math.pi
            angle_filter = self.get_filter(
                    center_of_mass, complex_plan, angle
                    ).to(x.device)
            radiuses[:, :, i, :] = torch.sqrt(
                (angle_filter * x).sum(list(range(2, len(x.shape) - 1))) * 2
                )
        return radiuses


class MomentGaussianRadiusExtractor(GaussianRadiusExtractor):
    """Class that extend the
    `diameter_learning.net.layers.GaussianRadiusExtractor`
    and compute radiuses between the center of mass and the border of a
    segmentation with an angle step of `\frac{2\pi}{nb\_radiuses}`

    :param moments: Moments used by the method to approximate the radiuses
    :param nb_radiuses: The number of radiuses
    :param sigma: The standard deviation of the gaussian used in the method
    :param epsilon: The parameter of the approximation
    """
    def __init__(
            self, moments: list = [0], nb_radiuses: int = 24,
            sigma: float = 0.1, epsilon=10**(-5)
            ):
        super().__init__(nb_radiuses, sigma, epsilon)
        self.moments = moments

    def get_centered_plan(
            self, center_of_mass: torch.Tensor,
            complex_plan: torch.Tensor,
            ) -> torch.Tensor:
        """Get a complex plan centered on the center of mass

        :param center_of_mass: A complex tensor of shape (nb, nf, nz)
            corresponding to the coordinates of the slice wise center of mass
            of the segmentation where nb is the batch size, nf the number of
            features, and nz the z dimension
        :param complex_plan: A complex tensor of shape (nx, ny) corresponding
            to complex plan coordinates where nx is the x dimension and ny
            the y dimension
        :return: A tensor of shape (nb, nf, nx, ny, nz) corresponding
            to complex plan coordinates where nb is the batch size, nf the
            number of features, nx the x dimension, ny the y dimension and nz
            the z dimension
        """
        # The different conditions represents the different case for the
        # angle due to the limit of the function torch.angle
        output_shape = (
            center_of_mass.shape[:-1] +
            complex_plan.shape +
            (center_of_mass.shape[-1],)
            )
        centered_plan = torch.complex(
                torch.zeros(output_shape), torch.zeros(output_shape)
                ).to(center_of_mass.device)
        for nb, nf, nz in product(
                range(center_of_mass.shape[0]),
                range(center_of_mass.shape[1]),
                range(center_of_mass.shape[-1])
                ):
            centered_plan[nb, nf, :, :, nz] = complex_plan - center_of_mass[
                nb, nf, nz
                ]
        return centered_plan

    def forward(self, x: torch.Tensor, center_of_mass: torch.Tensor) -> torch.Tensor:
        """Extension of the `torch.nn.Module.forward` method

        :param x: Tensor of shape (nb, nf, nx, ny, nz)
            where nb is the batch size, nf the number of features,
            nx the x dimension, ny the y dimension and nz the z dimension
        :param center_of_mass: A complex tensor of shape (nb, nf, nz)
            corresponding to the coordinates of the slice wise center of mass
            of the segmentation where nb is the batch size, nf the number of
            features, and nz the z dimension
        :return: Tensor of shape (nb, nf, nr, nz) where nb is the batch
            size, nf the number of features, nr the number of radiuses and nz
            the z dimension
        """
        # Generate complex plan
        index_array = torch.arange(
            x.shape[2] * x.shape[3]
            ).reshape(
                x.shape[2], x.shape[3]
                )
        complex_plan = torch.complex(
        (index_array % x.shape[3]).float(),
            (index_array // x.shape[3]).float()
            ).to(x.device)

        # Output
        radiuses = torch.zeros(
            (len(self.moments),) +
            x.shape[:2] +
            (self.nb_radiuses, x.shape[-1])
            ).to(x.device)

        centered_plan = self.get_centered_plan(
            center_of_mass, complex_plan
            ).to(x.device)

        # Apply the different integrals to get the radiuses
        for i, angle in enumerate(self.angles):
            angle = i * (2 * math.pi / self.nb_radiuses) - math.pi
            angle_filter = self.get_filter(
                center_of_mass, complex_plan, angle
                ).to(x.device)
            for j, moment in enumerate(self.moments):
                radiuses[j, :, :, i, :] = (
                    (moment + 2) *
                    (angle_filter * x * centered_plan.abs() ** moment).sum(
                        list(range(2, len(x.shape) - 1))
                        )
                    ) ** (1/(moment + 2))
        return radiuses


class VanillaDiameterExtractor(torch.nn.Module):
    """Class that extend the `torch.nn.Module` and compute the diameter
    of a segmentation

    :param nb_radiuses: The number of radiuses
    """
    def __init__(self, nb_radiuses: int = 24):
        super().__init__()
        self.nb_radiuses = nb_radiuses
        self.angles = torch.tensor(
            [
                i * (2 * math.pi / self.nb_radiuses) - math.pi
                for i in range(self.nb_radiuses)
                ]
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Extension of the `torch.nn.Module.forward` method

        :param x: Tensor of shape (nb, nf, nr, nz) where nb is the batch size,
            nf the number of features, nr the number of radiuses and
            nz the z dimension
        :return: Tensor of shape (nb, nf, nz) where nb is the batch size,
            nf the number of features, and nz the z dimension
        """
        nb, nf, _, nz = x.shape
        unitary_vector = torch.ones(1, self.nb_radiuses).to(x.device)
        diameters = torch.zeros(nb, nf, nz).to(x.device)
        for nb, nf, nz in product(
                range(x.shape[0]),
                range(x.shape[1]),
                range(x.shape[-1]),
                ):
            vectors = torch.complex(
                x[nb, nf, :, nz] * torch.cos(self.angles).to(x.device),
                x[nb, nf, :, nz] * torch.sin(self.angles).to(x.device)
                ).unsqueeze(0).to(x.device)
            distance_matrix = (
                torch.transpose(unitary_vector, 0, 1) * vectors -
                torch.transpose(vectors, 0, 1) * unitary_vector
                ).abs()
            diameters[nb, nf, nz] = distance_matrix.max()
        return diameters
