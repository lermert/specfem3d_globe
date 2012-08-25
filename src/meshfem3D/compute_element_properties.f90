!=====================================================================
!
!          S p e c f e m 3 D  G l o b e  V e r s i o n  5 . 1
!          --------------------------------------------------
!
!          Main authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!             and CNRS / INRIA / University of Pau, France
! (c) Princeton University and CNRS / INRIA / University of Pau
!                            April 2011
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
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

! compute several rheological and geometrical properties for a given spectral element
  subroutine compute_element_properties(ispec,iregion_code,idoubling, &
                         xstore,ystore,zstore,nspec,myrank,ABSORBING_CONDITIONS, &
                         RICB,RCMB,R670,RMOHO,RMOHO_FICTITIOUS_IN_MESHER,RTOPDDOUBLEPRIME, &
                         R600,R220,R771,R400,R120,R80,RMIDDLE_CRUST,ROCEAN, &
                         xelm,yelm,zelm,shape3D,rmin,rmax,rhostore,dvpstore, &
                         kappavstore,kappahstore,muvstore,muhstore,eta_anisostore, &
                         xixstore,xiystore,xizstore,etaxstore,etaystore,etazstore, &
                         gammaxstore,gammaystore,gammazstore,nspec_actually, &
                         c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
                         c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
                         c36store,c44store,c45store,c46store,c55store,c56store,c66store, &
                         nspec_ani,nspec_stacey,nspec_att,Qmu_store,tau_e_store,tau_s,T_c_source,&
                         rho_vp,rho_vs,ACTUALLY_STORE_ARRAYS,&
                         xigll,yigll,zigll,ispec_is_tiso)

  use meshfem3D_models_par

  implicit none

  !include "constants.h"

! correct number of spectral elements in each block depending on chunk type
  integer ispec,nspec,nspec_stacey

  logical ABSORBING_CONDITIONS,ACTUALLY_STORE_ARRAYS

  double precision RICB,RCMB,R670,RMOHO,RTOPDDOUBLEPRIME,R600,R220,R771,&
    R400,R120,R80,RMIDDLE_CRUST,ROCEAN,RMOHO_FICTITIOUS_IN_MESHER

! arrays with the mesh in double precision
  double precision xstore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision ystore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision zstore(NGLLX,NGLLY,NGLLZ,nspec)

! code for the four regions of the mesh
  integer iregion_code

! 3D shape functions and their derivatives
  double precision, dimension(NGNOD,NGLLX,NGLLY,NGLLZ) :: shape3D

  double precision, dimension(NGNOD) :: xelm,yelm,zelm

! parameters needed to store the radii of the grid points
! in the spherically symmetric Earth
  integer idoubling(nspec)
  double precision rmin,rmax

! for model density and anisotropy
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec) :: rhostore,dvpstore,kappavstore, &
    kappahstore,muvstore,muhstore,eta_anisostore

! the 21 coefficients for an anisotropic medium in reduced notation
  integer nspec_ani
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_ani) :: &
    c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
    c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
    c36store,c44store,c45store,c46store,c55store,c56store,c66store

! arrays with mesh parameters
  integer nspec_actually
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_actually) :: &
    xixstore,xiystore,xizstore,etaxstore,etaystore,etazstore,gammaxstore,gammaystore,gammazstore

! proc numbers for MPI
  integer myrank

! Stacey, indices for Clayton-Engquist absorbing conditions
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_stacey) :: rho_vp,rho_vs

! attenuation
  integer nspec_att
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_att) :: Qmu_store
  real(kind=CUSTOM_REAL), dimension(N_SLS,NGLLX,NGLLY,NGLLZ,nspec_att) :: tau_e_store
  double precision, dimension(N_SLS)                  :: tau_s
  double precision  T_c_source

  ! Parameters used to calculate Jacobian based upon 125 GLL points
  double precision:: xigll(NGLLX)
  double precision:: yigll(NGLLY)
  double precision:: zigll(NGLLZ)

  logical, dimension(nspec) :: ispec_is_tiso

  ! Parameter used to decide whether this element is in the crust or not
  logical:: elem_in_crust,elem_in_mantle
  ! flag for transverse isotropic elements
  logical:: elem_is_tiso

  ! add topography of the Moho *before* adding the 3D crustal velocity model so that the streched
  ! mesh gets assigned the right model values
  elem_in_crust = .false.
  elem_in_mantle = .false.
  elem_is_tiso = .false.
  if( iregion_code == IREGION_CRUST_MANTLE ) then
    if( CRUSTAL .and. CASE_3D ) then
      ! 3D crustal models
      if( idoubling(ispec) == IFLAG_CRUST &
        .or. idoubling(ispec) == IFLAG_220_80 &
        .or. idoubling(ispec) == IFLAG_80_MOHO ) then
        ! Stretch mesh to honor smoothed moho thickness from crust2.0

        ! differentiate between regional and global meshing
        if( REGIONAL_MOHO_MESH ) then
          call moho_stretching_honor_crust_reg(myrank, &
                              xelm,yelm,zelm,RMOHO_FICTITIOUS_IN_MESHER,&
                              R220,RMIDDLE_CRUST,elem_in_crust,elem_in_mantle)
        else
          call moho_stretching_honor_crust(myrank, &
                              xelm,yelm,zelm,RMOHO_FICTITIOUS_IN_MESHER,&
                              R220,RMIDDLE_CRUST,elem_in_crust,elem_in_mantle)
        endif
      else
        ! element below 220km
        ! sets element flag for mantle
        elem_in_mantle = .true.
      endif
    else
      ! 1D crust, no stretching
      ! sets element flags
      if( idoubling(ispec) == IFLAG_CRUST ) then
        elem_in_crust = .true.
      else
        elem_in_mantle = .true.
      endif
    endif

    ! sets transverse isotropic flag for elements in mantle
    if( TRANSVERSE_ISOTROPY ) then
      ! modifies tiso to have it for all mantle elements
      ! preferred for example, when using 1Dref (STW model)
      if( USE_FULL_TISO_MANTLE ) then
        ! all elements below the actual moho will be used for transverse isotropy
        ! note: this will increase the computation time by ~ 45 %
        if( elem_in_mantle ) then
          elem_is_tiso = .true.
        endif
      else if( REFERENCE_1D_MODEL == REFERENCE_MODEL_1DREF ) then
        ! transverse isotropic mantle between fictitious moho to 670km depth
        ! preferred for harvard (kustowski's) models using STW 1D reference, i.e.
        ! THREE_D_MODEL_S362ANI
        ! THREE_D_MODEL_S362WMANI
        ! THREE_D_MODEL_S29EA
        ! THREE_D_MODEL_GLL
        ! which show significant transverse isotropy also below 220km depth
        if( idoubling(ispec)==IFLAG_220_80 .or. idoubling(ispec)==IFLAG_80_MOHO &
          .or. idoubling(ispec)==IFLAG_670_220 ) then
          elem_is_tiso = .true.
        endif
      else if( idoubling(ispec)==IFLAG_220_80 .or. idoubling(ispec)==IFLAG_80_MOHO ) then
        ! default case for PREM reference models:
        ! models use only transverse isotropy between moho and 220 km depth
        elem_is_tiso = .true.
        ! checks mantle flag to be sure
        !if( elem_in_mantle .eqv. .false. ) stop 'error mantle flag confused between moho and 220'
      endif
    endif

  endif ! IREGION_CRUST_MANTLE

  ! sets element tiso flag
  ispec_is_tiso(ispec) = elem_is_tiso

  ! interpolates and stores GLL point locations
  call compute_element_GLL_locations(xelm,yelm,zelm,ispec,nspec, &
                                    xstore,ystore,zstore,shape3D)


  ! computes model's velocity/density/... values for the chosen Earth model
  call get_model(myrank,iregion_code,ispec,nspec,idoubling(ispec), &
                      kappavstore,kappahstore,muvstore,muhstore,eta_anisostore, &
                      rhostore,dvpstore,nspec_ani, &
                      c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
                      c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
                      c36store,c44store,c45store,c46store,c55store,c56store,c66store, &
                      nspec_stacey,rho_vp,rho_vs, &
                      xstore,ystore,zstore, &
                      rmin,rmax,RCMB,RICB,R670,RMOHO,RTOPDDOUBLEPRIME,R600,R220, &
                      R771,R400,R120,R80,RMIDDLE_CRUST,ROCEAN, &
                      tau_s,tau_e_store,Qmu_store,T_c_source, &
                      size(tau_e_store,2),size(tau_e_store,3),size(tau_e_store,4),size(tau_e_store,5), &
                      ABSORBING_CONDITIONS,elem_in_crust,elem_in_mantle)


  ! either use GLL points or anchor points to capture TOPOGRAPHY and ELLIPTICITY
  ! note:  using gll points to capture them results in a slightly more accurate mesh.
  !           however, it introduces more deformations to the elements which might lead to
  !           problems with the jacobian. using the anchors is therefore more robust.
  ! adds surface topography
  if( TOPOGRAPHY ) then
    if (idoubling(ispec)==IFLAG_CRUST .or. idoubling(ispec)==IFLAG_220_80 &
        .or. idoubling(ispec)==IFLAG_80_MOHO) then
      ! stretches mesh between surface and R220 accordingly
      if( USE_GLL ) then
        ! stretches every gll point accordingly
        call add_topography_gll(myrank,xstore,ystore,zstore,ispec,nspec,ibathy_topo,R220)
      else
        ! stretches anchor points only, interpolates gll points later on
        call add_topography(myrank,xelm,yelm,zelm,ibathy_topo,R220)
      endif
    endif
  endif

  ! adds topography on 410 km and 650 km discontinuity in model S362ANI
  if(THREE_D_MODEL == THREE_D_MODEL_S362ANI .or. THREE_D_MODEL == THREE_D_MODEL_S362WMANI &
    .or. THREE_D_MODEL == THREE_D_MODEL_S362ANI_PREM .or. THREE_D_MODEL == THREE_D_MODEL_S29EA) then
    if( USE_GLL ) then
      ! stretches every gll point accordingly
      call add_topography_410_650_gll(myrank,xstore,ystore,zstore,ispec,nspec,R220,R400,R670,R771, &
                                      numker,numhpa,numcof,ihpa,lmax,nylm, &
                                      lmxhpa,itypehpa,ihpakern,numcoe,ivarkern, &
                                      nconpt,iver,iconpt,conpt,xlaspl,xlospl,radspl, &
                                      coe,ylmcof,wk1,wk2,wk3,varstr)

    else
      ! stretches anchor points only, interpolates gll points later on
      call add_topography_410_650(myrank,xelm,yelm,zelm,R220,R400,R670,R771, &
                                      numker,numhpa,numcof,ihpa,lmax,nylm, &
                                      lmxhpa,itypehpa,ihpakern,numcoe,ivarkern, &
                                      nconpt,iver,iconpt,conpt,xlaspl,xlospl,radspl, &
                                      coe,ylmcof,wk1,wk2,wk3,varstr)
    endif
  endif

  ! these are placeholders:
  ! their corresponding subroutines subtopo_cmb() and subtopo_icb() are not implemented yet....
  ! must be done/supplied by the user; uncomment in case
  ! CMB topography
  !  if(THREE_D_MODEL == THREE_D_MODEL_S362ANI .and. (idoubling(ispec)==IFLAG_MANTLE_NORMAL &
  !     .or. idoubling(ispec)==IFLAG_OUTER_CORE_NORMAL)) &
  !           call add_topography_cmb(myrank,xelm,yelm,zelm,RTOPDDOUBLEPRIME,RCMB)

  ! ICB topography
  !  if(THREE_D_MODEL == THREE_D_MODEL_S362ANI .and. (idoubling(ispec)==IFLAG_OUTER_CORE_NORMAL &
  !     .or. idoubling(ispec)==IFLAG_INNER_CORE_NORMAL .or. idoubling(ispec)==IFLAG_MIDDLE_CENTRAL_CUBE &
  !     .or. idoubling(ispec)==IFLAG_BOTTOM_CENTRAL_CUBE .or. idoubling(ispec)==IFLAG_TOP_CENTRAL_CUBE &
  !     .or. idoubling(ispec)==IFLAG_IN_FICTITIOUS_CUBE)) &
  !           call add_topography_icb(myrank,xelm,yelm,zelm,RICB,RCMB)

  ! make the Earth elliptical
  if(ELLIPTICITY) then
    if( USE_GLL ) then
      ! make the Earth's ellipticity, use GLL points
      call get_ellipticity_gll(xstore,ystore,zstore,ispec,nspec,nspl,rspl,espl,espl2)
    else
      ! make the Earth's ellipticity, use element anchor points
      call get_ellipticity(xelm,yelm,zelm,nspl,rspl,espl,espl2)
    endif
  endif

  ! re-interpolates and creates the GLL point locations since the anchor points might have moved
  !
  ! note: velocity values associated for each GLL point will "move" along together with
  !          their associated points. however, we don't re-calculate the velocity model values since the
  !          models are/should be referenced with respect to a spherical Earth.
  if( .not. USE_GLL) &
    call compute_element_GLL_locations(xelm,yelm,zelm,ispec,nspec, &
                                      xstore,ystore,zstore,shape3D)

  ! updates jacobian
  call recalc_jacobian_gll3D(myrank,xstore,ystore,zstore,xigll,yigll,zigll,&
                                ispec,nspec,ACTUALLY_STORE_ARRAYS,&
                                xixstore,xiystore,xizstore,&
                                etaxstore,etaystore,etazstore,&
                                gammaxstore,gammaystore,gammazstore)

  end subroutine compute_element_properties

!
!-------------------------------------------------------------------------------------------------
!

  subroutine compute_element_GLL_locations(xelm,yelm,zelm,ispec,nspec, &
                                      xstore,ystore,zstore,shape3D)

  implicit none

  include "constants.h"

  integer ispec,nspec

  double precision xelm(NGNOD)
  double precision yelm(NGNOD)
  double precision zelm(NGNOD)

  double precision xstore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision ystore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision zstore(NGLLX,NGLLY,NGLLZ,nspec)

  double precision shape3D(NGNOD,NGLLX,NGLLY,NGLLZ)

  ! local parameters
  double precision xmesh,ymesh,zmesh
  integer i,j,k,ia

  do k=1,NGLLZ
    do j=1,NGLLY
      do i=1,NGLLX

        xmesh = ZERO
        ymesh = ZERO
        zmesh = ZERO

        ! interpolates the location using 3D shape functions
        do ia=1,NGNOD

          xmesh = xmesh + shape3D(ia,i,j,k)*xelm(ia)
          ymesh = ymesh + shape3D(ia,i,j,k)*yelm(ia)
          zmesh = zmesh + shape3D(ia,i,j,k)*zelm(ia)

        enddo

        ! stores mesh coordinates
        xstore(i,j,k,ispec) = xmesh
        ystore(i,j,k,ispec) = ymesh
        zstore(i,j,k,ispec) = zmesh

      enddo
    enddo
  enddo

  end subroutine compute_element_GLL_locations
