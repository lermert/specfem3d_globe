!=====================================================================
!
!          S p e c f e m 3 D  G l o b e  V e r s i o n  7 . 0
!          --------------------------------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================

!--------------------------------------------------------------------------------------------------
! Reference to Zheng model
!--------------------------------------------------------------------------------------------------

  module model_cjcrust_par

  ! parameters for northeast China & Sea of Japan crustal model
  ! from ambient seismic noise (Zheng et al., 2011)
  !       latitude : 28 - 48 degree N
  !       longitude: 124 - 146 degree N
  character(len=*), parameter :: PATHNAME_CJCRUST = 'DATA/cjcrust/CJcrust.txt'

  double precision, parameter :: CJCRUST_LON_MIN = 114.0d0
  double precision, parameter :: CJCRUST_LON_MAX = 156.0d0
  double precision, parameter :: CJCRUST_LAT_MIN =  16.0d0
  double precision, parameter :: CJCRUST_LAT_MAX =  60.0d0
  double precision, parameter :: CJCRUST_SAMPLE = 0.5d0
  double precision, parameter :: CJCRUST_SAMPLE_DEP = 0.05d0

  ! arrays for crustal model
  integer, parameter :: CJCRUST_NLON = 142, CJCRUST_NLAT = 90, CJCRUST_NDEP = 256
  ! integer, parameter :: CJCRUST_NLON = 85, CJCRUST_NLAT = 89, CJCRUST_NDEP = 1021

  double precision,dimension(:,:,:),allocatable :: lon_cj,lat_cj,moho_cj,depth_cj
  double precision,dimension(:,:,:),allocatable :: vp_cj,vs_cj,rho_cj

  ! smoothing
  logical, parameter :: flag_smooth_cjcrust = .false.  ! because: smoothing is already done during gridded model construction. 
  integer, parameter :: NTHETA_CJ = 4, NPHI_CJ = 20
  double precision, parameter :: cap_degree_CJ = 1.0d0

  end module model_cjcrust_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_cjcrust_broadcast()

  use constants
  use model_cjcrust_par

  implicit none

  integer :: ier

  ! allocates arrays for model
  allocate(lon_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           lat_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           moho_cj(CJCRUST_NLON,CJCRUST_NLAT,1), &
           depth_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           vp_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           vs_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           rho_cj(CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP), &
           stat=ier)
  if (ier /= 0 ) call exit_MPI(myrank,'Error allocating CJcrust arrays')

  ! read Zheng CRUST model on master
  if (myrank == 0) call read_cjcrust_model()

  ! broadcast Zheng CRUST model
  call bcast_all_dp(lon_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)
  call bcast_all_dp(lat_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)
  call bcast_all_dp(moho_cj,CJCRUST_NLON*CJCRUST_NLAT)
  call bcast_all_dp(depth_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)
  call bcast_all_dp(vp_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)
  call bcast_all_dp(vs_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)
  call bcast_all_dp(rho_cj,CJCRUST_NLON*CJCRUST_NLAT*CJCRUST_NDEP)

  end subroutine model_cjcrust_broadcast

!
!-------------------------------------------------------------------------------------------------
!

  subroutine read_cjcrust_model()

  use constants
  use model_cjcrust_par

  implicit none

  character(len=4),dimension(7) :: header
  double precision,dimension(7) :: tmp
  integer:: ilon,jlat,kdep,ier

  ! user output
  write(IMAIN,*)
  write(IMAIN,*) 'incorporating crustal model:'
  write(IMAIN,*) 'Ambient noise tomography model of northeast China and Japan'
  write(IMAIN,*) '  latitude  area: min/max = ',CJCRUST_LAT_MIN,'/',CJCRUST_LAT_MAX
  write(IMAIN,*) '  longitude area: min/max = ',CJCRUST_LON_MIN,'/',CJCRUST_LON_MAX
  write(IMAIN,*)

  open(unit=IIN,file=trim(PATHNAME_CJCRUST),status='old',action='read',iostat=ier)
  if (ier /= 0) then
    write(IMAIN,*) 'Error opening "', trim(PATHNAME_CJCRUST), '": ', ier
    call flush_IMAIN()
    ! stop
    call exit_MPI(0, 'Error model cjcrust')
  endif

  ! file format:
  !  LON   LAT   DEP   VP   VS   RHO   MOHO
  !  103.5   25.0   0.0   1.99505  1.99505   1.99505   25.0


  read(IIN,*) header
  ! print *,'header :',header

  do ilon = 1,CJCRUST_NLON
    do jlat = 1,CJCRUST_NLAT
      do kdep = 1,CJCRUST_NDEP
        ! data
        read(IIN,*) tmp
        lon_cj(ilon,jlat,kdep) = tmp(1)
        lat_cj(ilon,jlat,kdep) = tmp(2)
        depth_cj(ilon,jlat,kdep) = tmp(3)
        vp_cj(ilon,jlat,kdep) = tmp(4)
        vs_cj(ilon,jlat,kdep) = tmp(5)
        rho_cj(ilon,jlat,kdep) = tmp(6)
        ! print*, 'reading lon, lat, dep, vs', lon_cj(ilon,jlat,kdep), &
        ! lat_cj(ilon,jlat,kdep), depth_cj(ilon,jlat,kdep), vs_cj(ilon,jlat,kdep)
      enddo
    moho_cj(ilon,jlat,1) = tmp(7)
    enddo
  enddo
  close(IIN)
  close(IIN)

  end subroutine read_cjcrust_model

!
!-------------------------------------------------------------------------------------------------
!
  subroutine model_cjcrust(lat,lon,x,vpc,vsc,rhoc,mohoc,found_crust,elem_in_crust,point_in_area)

  use constants
  use model_cjcrust_par

  implicit none

  ! INPUT & OUTPUT
  double precision,intent(in) :: lat, lon, x
  double precision,intent(inout) :: vpc, vsc, rhoc, mohoc
  logical,intent(out) :: found_crust,point_in_area
  logical,intent(in) :: elem_in_crust

  ! local parameters
  integer:: ilon, jlat, kdep, k
  double precision :: vp, vs, rho, moho, depth
  double precision :: scaleval
  double precision,dimension(NTHETA_CJ*NPHI_CJ) :: x1,y1,weight
  double precision:: weightl

  !double precision:: min_sed
  ! moho threshold
  double precision:: minimum_moho_depth = 7.d0 / R_EARTH_KM

  ! initializes
  found_crust = .false.
  point_in_area = .false.

  ! min/max area:
  !
  ! CJcrust lat/lon range:     lat[25/ 26] / lon[103 / 104]
  !
  ! input value lat/lon given in range: lat[-90,90] / lon[-180,180]

  ! checks if anything to do
  if (lat < CJCRUST_LAT_MIN .or. lat > CJCRUST_LAT_MAX) return
  if (lon < CJCRUST_LON_MIN .or. lon > CJCRUST_LON_MAX) return
  point_in_area = .true.
  depth = R_EARTH_KM - x * R_EARTH_KM

  ! gets arrays
  if (.not. flag_smooth_cjcrust) then
    ! no smoothing
    call ilon_jlat_kdep(lon,lat,depth,ilon,jlat,kdep)
    moho = moho_cj(ilon,jlat,1)
    vp  = vp_cj(ilon,jlat,kdep)
    vs  = vs_cj(ilon,jlat,kdep)
    rho = rho_cj(ilon,jlat,kdep)
  else
    call cjcrust_smooth_base(lon,lat,x1,y1,weight)
    moho = ZERO
    vp  = ZERO
    vs  = ZERO
    rho = ZERO
    do k = 1,NTHETA_CJ*NPHI_CJ
      weightl = weight(k)
      call ilon_jlat_kdep(x1(k),y1(k),depth,ilon,jlat,kdep)
      moho  = moho+weightl*moho_cj(ilon,jlat,1)
      vp  = vp+weightl*vp_cj(ilon,jlat,kdep)
      vs  = vs+weightl*vs_cj(ilon,jlat,kdep)
      rho = rho+weightl*rho_cj(ilon,jlat,kdep)
      ! if (myrank == 0) then
       ! print*, 'vs before / after smoothing', vs_cj(ilon, jlat, kdep), vs
      ! endif
    enddo
  endif

  ! Hejun Zhu, delete moho thickness less than 7 km
  if (moho < minimum_moho_depth) then
    moho = minimum_moho_depth
  endif
  mohoc = moho / R_EARTH_KM

  if (depth < moho .or. elem_in_crust) then
    found_crust = .true.
    if (myrank == 0) then
      print*, lat, lon, x, vp, vs
    endif
  endif
  if (found_crust) then
    scaleval = dsqrt(PI*GRAV*RHOAV)
    vpc = vp*1000.d0/(R_EARTH*scaleval)
    vsc = vs*1000.d0/(R_EARTH*scaleval)
    rhoc = rho*1000.d0/RHOAV
  endif

  end subroutine model_cjcrust

!
!-------------------------------------------------------------------------------------------------
!
! not working as it should
  ! subroutine cjcrust_smooth_vert(z, z1, weight_depth)
  ! use constants
  ! use model_cjcrust_par, only: NZ_CJ, dep_km_CJ

  ! implicit none
  ! double precision:: z, total
  ! double precision, dimension(NZ_CJ):: z1, weight_depth
  ! integer:: n, nzall
  ! double precision:: dz
  ! ! smoothing in depth by averaging
  ! ! Maybe this is totally unnecessary, because the original model is spline-parametrized
  ! total = 0.0d0
  ! z1(:) = ZERO
  ! weight_depth(:) = ZERO
  ! nzall = (2 * NZ_CJ) + 1
  ! dz = dep_km_CJ / dble(nzall)

  ! print*, 'smoothing nzall dz z', nzall, dz, z

  ! do n=1,nzall
  !   z1(n) = abs(z - (NZ_CJ - n + 1) * dz)
  !   if (n <= NZ_CJ + 1) then
  !     weight_depth(n) = (n - 1) * (1.0d0 / (NZ_CJ * NZ_CJ))
  !   else
  !     weight_depth(n) = weight_depth(abs(n-nzall) + 1)
  !   endif
  !   total = total + weight_depth(n)
  !   ! if (myrank == 0) then
  !     ! print*, 'CJ weight depth, weight: ', z1(n), weight_depth(n)
  !   ! endif
  ! enddo
  ! if (myrank == 0) then
  !   print*, 'CJ cumulative weights for depth smoothing, ', total
  ! endif
  ! end subroutine cjcrust_smooth_vert

!
!-------------------------------------------------------------------------------------------------
!
! essentially from model_eucrust.f90 and model_epcrust.f90. Find comments in
! model_eucrust.f90

  subroutine cjcrust_smooth_base(x,y,x1,y1,weight)

  use constants
  use model_cjcrust_par, only: NTHETA_CJ,NPHI_CJ,cap_degree_CJ

  implicit none

  ! INPUT & OUTPUT
  double precision:: x, y
  double precision,dimension(NTHETA_CJ*NPHI_CJ):: x1,y1,weight

  ! INTERIOR
  double precision:: CAP,dtheta,dphi,cap_circumf,dweight,pi_over_nphi,total,wght
  double precision:: theta,phi,sint,cost,sinp,cosp
  double precision:: r_rot,theta_rot,phi_rot
  double precision,dimension(3,3):: rotation_matrix
  double precision,dimension(3):: xx,xc
  integer:: i,j,k,itheta,iphi

  x1(:) = ZERO
  y1(:) = ZERO
  weight(:) = ZERO

  if (cap_degree_CJ < TINYVAL) then
    print *, 'Error cap:', cap_degree_CJ
    print *, 'lat/lon:', x,y
    stop 'Error cap_degree too small'
  endif

  ! cap is a little cap rotated to the North Pole over which is smoothed
  CAP = cap_degree_CJ * DEGREES_TO_RADIANS
  ! theta controls the angle out from the North to which smoothin is done,
  ! i.e. basically the radius of the cap
  dtheta = 0.5d0*CAP/dble(NTHETA_CJ)
  ! phi goes once around the cap
  dphi = TWO_PI/dble(NPHI_CJ)
  ! R_EARTH normalized
  cap_circumf = TWO_PI*(1.0d0-dcos(CAP))
  dweight = CAP/dble(NTHETA_CJ)*dphi/cap_circumf
  pi_over_nphi = PI/dble(NPHI_CJ)

  phi = x*DEGREES_TO_RADIANS
  theta = (90.0d0-y)*DEGREES_TO_RADIANS

  sint = dsin(theta)
  cost = dcos(theta)
  sinp = dsin(phi)
  cosp = dcos(phi)

  ! matrix to go from North pole to location of interest (lat, lon)
  rotation_matrix(1,1) = cosp*cost
  rotation_matrix(1,2) = -sinp
  rotation_matrix(1,3) = cosp*sint
  rotation_matrix(2,1) = sinp*cost
  rotation_matrix(2,2) = cosp
  rotation_matrix(2,3) = sinp*sint
  rotation_matrix(3,1) = -sint
  rotation_matrix(3,2) = ZERO
  rotation_matrix(3,3) = cost

  i = 0
  total = 0.0d0
  do itheta = 1,NTHETA_CJ
    theta = dble(2*itheta-1)*dtheta
    cost = dcos(theta)
    sint = dsin(theta)
    wght = sint*dweight
    do iphi = 1,NPHI_CJ
      i = i+1
      weight(i) = wght

      total = total+weight(i)
      phi = dble(2*iphi-1)*pi_over_nphi
      cosp = dcos(phi)
      sinp = dsin(phi)

      xc(1) = sint*cosp
      xc(2) = sint*sinp
      xc(3) = cost
      do j = 1,3
        xx(j) = 0.0d0
        do k = 1,3
          xx(j)=xx(j)+rotation_matrix(j,k)*xc(k)
        enddo
      enddo
      call xyz_2_rthetaphi_dble(xx(1),xx(2),xx(3),r_rot,theta_rot,phi_rot)
      call reduce(theta_rot,phi_rot)
      x1(i) = phi_rot*RADIANS_TO_DEGREES
      y1(i) = (PI_OVER_TWO-theta_rot)*RADIANS_TO_DEGREES
      if (x1(i) > 180.d0) x1(i) = x1(i)-360.d0
    enddo
  enddo

  if (abs(total-1.0d0) > 0.001d0) then
    print *,'Error cap:',total,cap_degree_CJ
    stop
  endif

  end subroutine cjcrust_smooth_base

!
!-------------------------------------------------------------------------------------------------
!

  subroutine ilon_jlat_kdep(lon,lat,dep,ilon,jlat,kdep)

  use constants
  use model_cjcrust_par, only: &
    CJCRUST_LON_MIN,CJCRUST_LAT_MIN,CJCRUST_SAMPLE, &
    CJCRUST_NLON,CJCRUST_NLAT,CJCRUST_NDEP,CJCRUST_SAMPLE_DEP

  implicit none

  double precision:: lon,lat,dep
  integer:: ilon,jlat,kdep

  ilon = nint((lon-CJCRUST_LON_MIN)/CJCRUST_SAMPLE)+1
  jlat = nint((lat-CJCRUST_LAT_MIN)/CJCRUST_SAMPLE)+1
  kdep = nint(dep / CJCRUST_SAMPLE_DEP) +1

  if (ilon < 1) ilon = 1
  if (ilon > CJCRUST_NLON) ilon = CJCRUST_NLON
  if (jlat < 1) jlat = 1
  if (jlat > CJCRUST_NLAT) jlat = CJCRUST_NLAT
  if (kdep < 1) kdep = 1
  if (kdep > CJCRUST_NDEP) kdep = CJCRUST_NDEP

  end subroutine ilon_jlat_kdep
