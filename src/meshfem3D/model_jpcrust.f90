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
! Reference to Nishida et al. (2008) model
!--------------------------------------------------------------------------------------------------

  module model_jpcrust_par

  ! parameters for Japan islands crustal model
  ! from ambient noise tomography (Nishida et al., 2008, JGR)
  !       model extent is limited to Japan islands
  !       model is buffered by 1-degree smoothed crust1.0
  character(len=*), parameter :: PATHNAME_JPCRUST = 'DATA/jpcrust/jpcrust.txt'

  double precision, parameter :: JPCRUST_LON_MIN = 99.99d0
  double precision, parameter :: JPCRUST_LON_MAX = 170.1d0
  double precision, parameter :: JPCRUST_LAT_MIN =  15.99d0
  double precision, parameter :: JPCRUST_LAT_MAX =  60.1d0
  double precision, parameter :: JPCRUST_SAMPLE = 0.2d0
  double precision, parameter :: JPCRUST_SAMPLE_DEP = 1.0d0

  ! arrays for crustal model
  integer, parameter :: JPCRUST_NLON = 352, JPCRUST_NLAT = 222, &
  JPCRUST_NDEP = 60

  double precision,dimension(:,:,:),allocatable :: lon_jp,lat_jp
  double precision, dimension(:, :, :), allocatable :: moho_jp,depth_jp
  double precision,dimension(:,:,:),allocatable :: vp_jp,vs_jp,rho_jp

  ! smoothing
  logical, parameter :: flag_smooth_jpcrust = .false.
  ! because: smoothing is already done during gridded model construction.

  end module model_jpcrust_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_jpcrust_broadcast()

  use constants
  use model_jpcrust_par

  implicit none

  integer :: ier

  ! allocates arrays for model
  allocate(lon_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           lat_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           moho_jp(JPCRUST_NLON,JPCRUST_NLAT,1), &
           depth_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           vp_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           vs_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           rho_jp(JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP), &
           stat=ier)
  if (ier /= 0 ) call exit_MPI(myrank,'Error allocating JPcrust arrays')

  ! read model on master
  if (myrank == 0) call read_jpcrust_model()

  ! broadcast model
  call bcast_all_dp(lon_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)
  call bcast_all_dp(lat_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)
  call bcast_all_dp(moho_jp,JPCRUST_NLON*JPCRUST_NLAT)
  call bcast_all_dp(depth_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)
  call bcast_all_dp(vp_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)
  call bcast_all_dp(vs_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)
  call bcast_all_dp(rho_jp,JPCRUST_NLON*JPCRUST_NLAT*JPCRUST_NDEP)

  end subroutine model_jpcrust_broadcast

!
!-------------------------------------------------------------------------------------------------
!

  subroutine read_jpcrust_model()

  use constants
  use model_jpcrust_par

  implicit none

  character(len=4),dimension(7) :: header
  double precision,dimension(7) :: tmp
  integer:: ilon,jlat,kdep,ier

  ! user output
  write(IMAIN,*)
  write(IMAIN,*) 'incorporating crustal model:'
  write(IMAIN,*) 'Ambient noise tomography model of Japan + crust1.0'
  write(IMAIN,*) '  latitude  area: min/max = ',JPCRUST_LAT_MIN,'/',JPCRUST_LAT_MAX
  write(IMAIN,*) '  longitude area: min/max = ',JPCRUST_LON_MIN,'/',JPCRUST_LON_MAX
  write(IMAIN,*)

  open(unit=IIN,file=trim(PATHNAME_JPCRUST),status='old',action='read',iostat=ier)
  if (ier /= 0) then
    write(IMAIN,*) 'Error opening "', trim(PATHNAME_JPCRUST), '": ', ier
    call flush_IMAIN()
    ! stop
    call exit_MPI(0, 'Error model jpcrust')
  endif

  ! file format:
  !  LON   LAT   DEP   VP   VS   RHO   MOHO
  !  103.5   25.0   0.0   1.99505  1.99505   1.99505   25.0


  read(IIN,*) header
  ! print *,'header :',header

  do ilon = 1,JPCRUST_NLON
    do jlat = 1,JPCRUST_NLAT
      do kdep = 1,JPCRUST_NDEP
        ! data
        read(IIN,*) tmp
        lon_jp(ilon,jlat,kdep) = tmp(1)
        lat_jp(ilon,jlat,kdep) = tmp(2)
        depth_jp(ilon,jlat,kdep) = tmp(3)
        vp_jp(ilon,jlat,kdep) = tmp(4)
        vs_jp(ilon,jlat,kdep) = tmp(5)
        rho_jp(ilon,jlat,kdep) = tmp(6)
        ! print*, 'reading lon, lat, dep, vs', lon_jp(ilon,jlat,kdep), &
        ! lat_jp(ilon,jlat,kdep), depth_jp(ilon,jlat,kdep), vs_jp(ilon,jlat,kdep)
      enddo
    moho_jp(ilon,jlat,1) = tmp(7)
    enddo
  enddo
  close(IIN)
  close(IIN)

  end subroutine read_jpcrust_model

!
!-------------------------------------------------------------------------------------------------
!
  subroutine model_jpcrust(lat,lon,x,vpc,vsc,rhoc,mohoc,found_crust,elem_in_crust,point_in_area)

  use constants
  use model_jpcrust_par

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
  ! double precision,dimension(NTHETA_JP*NPHI_JP) :: x1,y1,weight
  ! double precision:: weightl

  !double precision:: min_sed
  ! moho threshold
  double precision:: minimum_moho_depth = 7.d0 / R_EARTH_KM

  ! initializes
  found_crust = .false.
  point_in_area = .false.

  ! min/max area:
  !
  ! JPcrust lat/lon range:     lat[16/ 60] / lon[114 / 156]
  !
  ! input value lat/lon given in range: lat[-90,90] / lon[-180,180]

  ! checks if anything to do
  if (lat < JPCRUST_LAT_MIN .or. lat > JPCRUST_LAT_MAX) then
    print*, 'OOA'
    return
  endif
  if (lon < JPCRUST_LON_MIN .or. lon > JPCRUST_LON_MAX) then
    print*, 'OOA'
    return
  endif
  point_in_area = .true.
  depth = R_EARTH_KM - x * R_EARTH_KM

  ! gets arrays
  if (.not. flag_smooth_jpcrust) then
    ! no smoothing
    call ilon_jlat_kdep_jp(lon,lat,depth,ilon,jlat,kdep)
    moho = moho_jp(ilon,jlat,1)
    vp  = vp_jp(ilon,jlat,kdep)
    vs  = vs_jp(ilon,jlat,kdep)
    rho = rho_jp(ilon,jlat,kdep)
  else
    call exit_MPI(myrank, &
          'Error: Smoothing of Japan ambient noise tomo model needs to be done during preparation')
  endif

  ! Hejun Zhu, delete moho thickness less than 7 km
  if (moho < minimum_moho_depth) then
    moho = minimum_moho_depth
  endif
  mohoc = moho / R_EARTH_KM
  
  if (depth < moho .or. elem_in_crust) then
    found_crust = .true.
    if (myrank == 0) then
      print*, lat, lon, x, vp, vs, moho
    endif
  endif
  if (found_crust) then
    scaleval = dsqrt(PI*GRAV*RHOAV)
    vpc = vp*1000.d0/(R_EARTH*scaleval)
    vsc = vs*1000.d0/(R_EARTH*scaleval)
    rhoc = rho*1000.d0/RHOAV
  endif

  end subroutine model_jpcrust


!
!-------------------------------------------------------------------------------------------------
!

  subroutine ilon_jlat_kdep_jp(lon,lat,dep,ilon,jlat,kdep)

  use constants
  use model_jpcrust_par, only: &
    JPCRUST_LON_MIN,JPCRUST_LAT_MIN,JPCRUST_SAMPLE, &
    JPCRUST_NLON,JPCRUST_NLAT,JPCRUST_NDEP,JPCRUST_SAMPLE_DEP

  implicit none

  double precision:: lon,lat,dep
  integer:: ilon,jlat,kdep

  ilon = nint((lon-JPCRUST_LON_MIN)/JPCRUST_SAMPLE)+1
  jlat = nint((lat-JPCRUST_LAT_MIN)/JPCRUST_SAMPLE)+1
  kdep = nint(dep / JPCRUST_SAMPLE_DEP) +1

  if (ilon < 1) ilon = 1
  if (ilon > JPCRUST_NLON) ilon = JPCRUST_NLON
  if (jlat < 1) jlat = 1
  if (jlat > JPCRUST_NLAT) jlat = JPCRUST_NLAT
  if (kdep < 1) kdep = 1
  if (kdep > JPCRUST_NDEP) kdep = JPCRUST_NDEP

  end subroutine ilon_jlat_kdep_jp
