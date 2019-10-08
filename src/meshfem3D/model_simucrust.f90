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
! Reference to Simute model
!--------------------------------------------------------------------------------------------------

  module model_simucrust_par

  ! parameters for Korea, NE Japan, Sea of Japan & Japanese islands crustal model
  ! from full waveform inversion (Simute et al., 2016)
  !       latitude of model : 15 - 60 degree N
  !       longitude of model: 115 - 160 degree N
  character(len=*), parameter :: PATHNAME_SIMUCRUST = 'DATA/simucrust/simucrust.txt'

  double precision, parameter :: SIMUCRUST_LON_MIN = 100.0d0
  double precision, parameter :: SIMUCRUST_LON_MAX = 170.0d0
  double precision, parameter :: SIMUCRUST_LAT_MIN =  16.0d0
  double precision, parameter :: SIMUCRUST_LAT_MAX =  60.0d0
  double precision, parameter :: SIMUCRUST_SAMPLE = 0.25d0
  double precision, parameter :: SIMUCRUST_SAMPLE_DEP = 1.0d0

  ! arrays for crustal model
  integer, parameter :: SIMUCRUST_NLON = 281, SIMUCRUST_NLAT = 177, &
  SIMUCRUST_NDEP = 56

  double precision,dimension(:,:,:),allocatable :: lon_simu,lat_simu
  double precision, dimension(:, :, :), allocatable :: moho_simu,depth_simu
  double precision,dimension(:,:,:),allocatable :: vp_simu,vs_simu,rho_simu

  ! smoothing
  logical, parameter :: flag_smooth_simucrust = .false.
  ! because: smoothing is already done during gridded model construction.

  end module model_simucrust_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_simucrust_broadcast()

  use constants
  use model_simucrust_par

  implicit none

  integer :: ier

  ! allocates arrays for model
  allocate(lon_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           lat_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           moho_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,1), &
           depth_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           vp_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           vs_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           rho_simu(SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP), &
           stat=ier)
  if (ier /= 0 ) call exit_MPI(myrank,'Error allocating SIMUcrust arrays')

  ! read Zheng CRUST model on master
  if (myrank == 0) call read_simucrust_model()

  ! broadcast Zheng CRUST model
  call bcast_all_dp(lon_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)
  call bcast_all_dp(lat_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)
  call bcast_all_dp(moho_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT)
  call bcast_all_dp(depth_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)
  call bcast_all_dp(vp_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)
  call bcast_all_dp(vs_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)
  call bcast_all_dp(rho_simu,SIMUCRUST_NLON*SIMUCRUST_NLAT*SIMUCRUST_NDEP)

  end subroutine model_simucrust_broadcast

!
!-------------------------------------------------------------------------------------------------
!

  subroutine read_simucrust_model()

  use constants
  use model_simucrust_par

  implicit none

  character(len=4),dimension(7) :: header
  double precision,dimension(7) :: tmp
  integer:: ilon,jlat,kdep,ier

  ! user output
  write(IMAIN,*)
  write(IMAIN,*) 'incorporating crustal model:'
  write(IMAIN,*) 'FWI model of northeast China, Korea, Japan'
  write(IMAIN,*) '  latitude  area: min/max = ',SIMUCRUST_LAT_MIN,'/',SIMUCRUST_LAT_MAX
  write(IMAIN,*) '  longitude area: min/max = ',SIMUCRUST_LON_MIN,'/',SIMUCRUST_LON_MAX
  write(IMAIN,*)

  open(unit=IIN,file=trim(PATHNAME_SIMUCRUST),status='old',action='read',iostat=ier)
  if (ier /= 0) then
    write(IMAIN,*) 'Error opening "', trim(PATHNAME_SIMUCRUST), '": ', ier
    call flush_IMAIN()
    ! stop
    call exit_MPI(0, 'Error model simucrust')
  endif

  ! file format:
  !  LON   LAT   DEP   VP   VS   RHO   MOHO
  !  103.5   25.0   0.0   1.99505  1.99505   1.99505   25.0


  read(IIN,*) header
  ! print *,'header :',header

  do ilon = 1,SIMUCRUST_NLON
    do jlat = 1,SIMUCRUST_NLAT
      do kdep = 1,SIMUCRUST_NDEP
        ! data
        read(IIN,*) tmp
        lon_simu(ilon,jlat,kdep) = tmp(1)
        lat_simu(ilon,jlat,kdep) = tmp(2)
        depth_simu(ilon,jlat,kdep) = tmp(3)
        vp_simu(ilon,jlat,kdep) = tmp(4)
        vs_simu(ilon,jlat,kdep) = tmp(5)
        rho_simu(ilon,jlat,kdep) = tmp(6)
        ! print*, 'reading lon, lat, dep, vs', lon_simu(ilon,jlat,kdep), &
        ! lat_simu(ilon,jlat,kdep), depth_simu(ilon,jlat,kdep), vs_simu(ilon,jlat,kdep)
      enddo
    moho_simu(ilon,jlat,1) = tmp(7)
    enddo
  enddo
  close(IIN)
  close(IIN)

  end subroutine read_simucrust_model

!
!-------------------------------------------------------------------------------------------------
!
  subroutine model_simucrust(lat,lon,x,vpc,vsc,rhoc,mohoc,found_crust,elem_in_crust,point_in_area)

  use constants
  use model_simucrust_par

  implicit none

  ! INPUT & OUTPUT
  double precision,intent(in) :: lat, lon, x
  double precision,intent(inout) :: vpc, vsc, rhoc, mohoc
  logical,intent(out) :: found_crust,point_in_area
  logical,intent(in) :: elem_in_crust

  ! local parameters
  integer:: ilon, jlat, kdep
  ! integer :: k
  double precision :: vp, vs, rho, moho, depth
  double precision :: scaleval
  ! double precision,dimension(NTHETA_SIMU*NPHI_SIMU) :: x1,y1,weight
  ! double precision:: weightl

  !double precision:: min_sed
  ! moho threshold
  double precision:: minimum_moho_depth = 7.d0 / R_EARTH_KM

  ! initializes
  found_crust = .false.
  point_in_area = .false.

  ! min/max area:
  !
  ! SIMUcrust lat/lon range:     lat[16/ 60] / lon[114 / 156]
  !
  ! input value lat/lon given in range: lat[-90,90] / lon[-180,180]

  ! checks if anything to do
  if (lat < SIMUCRUST_LAT_MIN .or. lat > SIMUCRUST_LAT_MAX) then
    print*, 'OOA'
    return
  endif
  if (lon < SIMUCRUST_LON_MIN .or. lon > SIMUCRUST_LON_MAX) then
    print*, 'OOA'
    return
  endif
  point_in_area = .true.
  depth = R_EARTH_KM - x * R_EARTH_KM

  ! gets arrays
  if (.not. flag_smooth_simucrust) then
    ! no smoothing
    call ilon_jlat_kdep_simu(lon,lat,depth,ilon,jlat,kdep)
    moho = moho_simu(ilon,jlat,1)
    vp  = vp_simu(ilon,jlat,kdep)
    vs  = vs_simu(ilon,jlat,kdep)
    rho = rho_simu(ilon,jlat,kdep)
  else
    call exit_MPI(myrank, &
          'Error: Smoothing of Simute model needs to be done during preparation')
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

  end subroutine model_simucrust


!
!-------------------------------------------------------------------------------------------------
!

  subroutine ilon_jlat_kdep_simu(lon,lat,dep,ilon,jlat,kdep)

  use constants
  use model_simucrust_par, only: &
    SIMUCRUST_LON_MIN,SIMUCRUST_LAT_MIN,SIMUCRUST_SAMPLE, &
    SIMUCRUST_NLON,SIMUCRUST_NLAT,SIMUCRUST_NDEP,SIMUCRUST_SAMPLE_DEP

  implicit none

  double precision:: lon,lat,dep
  integer:: ilon,jlat,kdep

  ilon = nint((lon-SIMUCRUST_LON_MIN)/SIMUCRUST_SAMPLE)+1
  jlat = nint((lat-SIMUCRUST_LAT_MIN)/SIMUCRUST_SAMPLE)+1
  kdep = nint(dep / SIMUCRUST_SAMPLE_DEP) +1

  if (ilon < 1) ilon = 1
  if (ilon > SIMUCRUST_NLON) ilon = SIMUCRUST_NLON
  if (jlat < 1) jlat = 1
  if (jlat > SIMUCRUST_NLAT) jlat = SIMUCRUST_NLAT
  if (kdep < 1) kdep = 1
  if (kdep > SIMUCRUST_NDEP) kdep = SIMUCRUST_NDEP

  end subroutine ilon_jlat_kdep_simu
